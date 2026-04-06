"""
physics.py — Aircraft Longitudinal Dynamics (2D)

Models a simplified fixed-wing aircraft in the vertical plane.
State vector: [V, gamma, h]
  V     : airspeed (m/s)
  gamma : flight-path angle (rad)  — positive = climbing
  h     : altitude (m)

Aerodynamic model
-----------------
  CL = 2π·α                    (thin-airfoil approximation)
  CD = CD0 + k·CL²             (polar parabolic drag)
  L  = ½·ρ·V²·S·CL
  D  = ½·ρ·V²·S·CD

Equations of motion (point-mass, no rotational inertia)
-------------------------------------------------------
  dV/dt     = (T·cos(α) - D - m·g·sin(γ)) / m
  dγ/dt     = (L + T·sin(α) - m·g·cos(γ)) / (m·V)
  dh/dt     =  V·sin(γ)

where
  α (angle of attack) = θ (pitch angle) - γ
  θ is a control input here (we set pitch to command lift indirectly)
"""

import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
G = 9.81          # gravitational acceleration  (m/s²)
RHO_SL = 1.225    # sea-level air density       (kg/m³)

# ── Aircraft parameters (generic light aircraft) ──────────────────────────────
AIRCRAFT = dict(
    m  = 5000.0,   # mass                        (kg)
    S  = 25.0,     # wing reference area         (m²)
    CD0 = 0.02,    # zero-lift drag coefficient  (–)
    k   = 0.04,    # induced drag factor  CD=CD0+k*alpha² (–)
    T_max = 80000, # max thrust                  (N)
    T_min = 0.0,   # min thrust (idle)           (N)
)


def air_density(h: float) -> float:
    """
    International Standard Atmosphere (troposphere up to ~11 km).
    ρ(h) = ρ_SL · (1 - L·h/T0)^(g/(R·L))
    Simplified as exponential decay for altitudes up to ~10 km.
    """
    # Simple exponential model (good enough for 0–8 km)
    return RHO_SL * np.exp(-h / 8500.0)


def aerodynamics(V: float, alpha: float, h: float, params: dict):
    """
    Compute lift and drag forces.

    Parameters
    ----------
    V     : airspeed (m/s)
    alpha : angle of attack (rad)
    h     : altitude (m)
    params: aircraft parameter dict

    Returns
    -------
    L, D : lift and drag (N)
    """
    rho = air_density(h)
    q   = 0.5 * rho * V**2          # dynamic pressure (Pa)
    S   = params['S']

    # Aerodynamic coefficients
    CL = 2.0 * np.pi * alpha         # thin-airfoil theory
    CD = params['CD0'] + params['k'] * alpha**2

    L = q * S * CL
    D = q * S * CD
    return L, D


def equations_of_motion(t: float, state: np.ndarray,
                         theta: float, thrust: float,
                         params: dict,
                         wind_speed: float = 0.0,
                         noise_std: float = 0.0) -> np.ndarray:
    """
    Returns time derivatives of the state vector.

    Parameters
    ----------
    t         : current time (s)  — required by solve_ivp signature
    state     : [V, gamma, h]
    theta     : pitch angle (rad) — commanded by autopilot / user
    thrust    : engine thrust (N)
    params    : aircraft parameters dict
    wind_speed: horizontal wind (m/s), positive = headwind
    noise_std : std-dev of Gaussian sensor noise added to state

    Returns
    -------
    [dV/dt, dgamma/dt, dh/dt]
    """
    V, gamma, h = state

    # Guard against zero / negative speed
    V = max(V, 1.0)

    # Optional wind disturbance — modeled as effective airspeed change
    V_eff = V + wind_speed * np.cos(gamma)   # wind along flight path
    V_eff = max(V_eff, 1.0)

    # Angle of attack
    alpha = theta - gamma

    # Clamp alpha to realistic range (±20°) to prevent model blow-up
    alpha = np.clip(alpha, -np.radians(20), np.radians(20))

    m = params['m']
    g = G

    L, D = aerodynamics(V_eff, alpha, h, params)

    # Equations of motion
    dV_dt     = (thrust * np.cos(alpha) - D - m * g * np.sin(gamma)) / m
    dgamma_dt = (L + thrust * np.sin(alpha) - m * g * np.cos(gamma)) / (m * V_eff)
    dh_dt     = V * np.sin(gamma)

    # Optional sensor noise on derivatives (simulates real-world IMU noise)
    if noise_std > 0:
        dV_dt     += np.random.normal(0, noise_std * 0.1)
        dgamma_dt += np.random.normal(0, noise_std * 0.001)
        dh_dt     += np.random.normal(0, noise_std * 0.05)

    return np.array([dV_dt, dgamma_dt, dh_dt])


def rk4_step(f, t: float, state: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    """
    Classical 4th-order Runge-Kutta integration step.

    dy/dt = f(t, y)   →   y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
    """
    k1 = f(t,          state,              **kwargs)
    k2 = f(t + dt/2,   state + dt/2 * k1, **kwargs)
    k3 = f(t + dt/2,   state + dt/2 * k2, **kwargs)
    k4 = f(t + dt,     state + dt   * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class AircraftSim:
    """
    Self-contained aircraft simulation object.
    Owns state, time, and advances physics by one timestep.
    """

    def __init__(self, V0=80.0, gamma0=0.0, h0=1000.0,
                 dt=0.05, params=None):
        """
        Parameters
        ----------
        V0, gamma0, h0 : initial airspeed (m/s), FPA (rad), altitude (m)
        dt             : fixed timestep (s)
        params         : aircraft parameter dict (defaults to AIRCRAFT)
        """
        self.params   = params or AIRCRAFT.copy()
        self.dt       = dt
        self.t        = 0.0
        self.state    = np.array([V0, gamma0, h0], dtype=float)

        # Control inputs (set externally each step)
        self.thrust   = 0.0    # N
        self.theta    = 0.0    # rad

        # Environment
        self.wind_speed = 0.0  # m/s
        self.noise_std  = 0.0  # dimensionless scale

    @property
    def V(self)     -> float: return self.state[0]
    @property
    def gamma(self) -> float: return self.state[1]
    @property
    def h(self)     -> float: return self.state[2]
    @property
    def alpha(self) -> float: return self.theta - self.gamma

    def step(self):
        """Advance simulation by one dt using RK4."""
        self.state = rk4_step(
            equations_of_motion,
            self.t,
            self.state,
            self.dt,
            theta      = self.theta,
            thrust     = self.thrust,
            params     = self.params,
            wind_speed = self.wind_speed,
            noise_std  = self.noise_std,
        )
        # Safety floor — aircraft can't go below ground
        if self.state[2] < 0:
            self.state[2] = 0.0
            self.state[1] = max(self.state[1], 0.0)

        self.t += self.dt