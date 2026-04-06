"""
control.py — PID Autopilot for Altitude Hold

The controller receives altitude error (h_ref - h) and outputs a pitch command (theta)
which the physics engine converts to lift via angle of attack.

Architecture
------------
  error = h_ref - h
  theta_cmd = Kp·e + Ki·∫e dt + Kd·(de/dt)

The pitch command is then used to derive angle of attack:
  alpha = theta_cmd - gamma

Thrust is set proportionally to maintain cruise speed (feed-forward + PI).
"""

import numpy as np


class PIDController:
    """
    Generic PID controller with anti-windup and output saturation.

    u(t) = Kp·e(t) + Ki·∫e dt + Kd·(de/dt)
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 output_min: float = -np.inf,
                 output_max: float =  np.inf,
                 integral_limit: float = 1e6):
        """
        Parameters
        ----------
        Kp, Ki, Kd     : PID gains
        output_min/max : saturation limits on controller output
        integral_limit : anti-windup clamp on integral term
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit

        # Internal state
        self._integral    = 0.0
        self._prev_error  = None
        self._prev_output = 0.0

    def reset(self):
        """Zero out integrator and derivative memory."""
        self._integral   = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        """
        Compute PID output given current error and timestep.

        Parameters
        ----------
        error : setpoint minus measurement
        dt    : elapsed time since last call (s)

        Returns
        -------
        output : controller output, clamped to [output_min, output_max]
        """
        if dt <= 0:
            return self._prev_output

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup clamp
        self._integral += error * dt
        self._integral  = np.clip(self._integral,
                                  -self.integral_limit, self.integral_limit)
        I = self.Ki * self._integral

        # Derivative term (backward difference; avoid derivative kick on setpoint change)
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative

        self._prev_error = error

        # Sum and saturate
        output = np.clip(P + I + D, self.output_min, self.output_max)
        self._prev_output = output
        return output

    def set_gains(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


class AltitudeHoldAutopilot:
    """
    Two-loop autopilot for altitude hold:

    Outer loop — altitude controller:
        error  = h_ref - h
        output = desired pitch angle theta_cmd  (rad)

    Inner loop — thrust controller:
        Maintains near-constant airspeed via thrust adjustment.
        Feed-forward: thrust = m*g (weight support at level flight)
        Feedback: proportional correction on speed error

    This mirrors real aircraft autopilot design (AFCS).
    """

    def __init__(self, Kp_alt=0.02, Ki_alt=0.001, Kd_alt=0.15,
                 Kp_spd=200.0,
                 T_min=0.0, T_max=80000.0,
                 V_ref=80.0, m=5000.0):
        """
        Parameters
        ----------
        Kp/Ki/Kd_alt : altitude PID gains
        Kp_spd       : speed proportional gain (N per m/s error)
        T_min/T_max  : thrust saturation (N)
        V_ref        : target airspeed (m/s)
        m            : aircraft mass (kg) for feed-forward
        """
        self.V_ref = V_ref
        self.m     = m
        self.T_min = T_min
        self.T_max = T_max

        # Altitude → pitch PID
        self.alt_pid = PIDController(
            Kp=Kp_alt, Ki=Ki_alt, Kd=Kd_alt,
            output_min=-np.radians(15),   # pitch limits ±15°
            output_max= np.radians(15),
        )

        # Speed → thrust P controller (no integral needed with FF)
        self.Kp_spd = Kp_spd

        # Feed-forward thrust: weight component for level flight
        self._ff_thrust = m * 9.81 * 0.02   # ≈ drag at cruise alpha

    def reset(self):
        self.alt_pid.reset()

    def compute(self, h: float, h_ref: float,
                V: float, gamma: float, dt: float):
        """
        Compute thrust and pitch commands.

        Parameters
        ----------
        h     : current altitude (m)
        h_ref : target altitude (m)
        V     : current airspeed (m/s)
        gamma : current flight-path angle (rad)
        dt    : timestep (s)

        Returns
        -------
        thrust : commanded thrust (N)
        theta  : commanded pitch angle (rad)
        """
        # ── Outer loop: altitude → pitch ──────────────────────────────────────
        alt_error = h_ref - h
        theta_cmd = self.alt_pid.update(alt_error, dt)

        # ── Inner loop: speed → thrust ────────────────────────────────────────
        # Feed-forward: thrust needed to overcome gravity component along path
        ff = self.m * 9.81 * np.sin(gamma) + self._ff_thrust
        # Feedback: correct speed deviation
        spd_error = self.V_ref - V
        thrust = ff + self.Kp_spd * spd_error
        thrust = np.clip(thrust, self.T_min, self.T_max)

        return thrust, theta_cmd

    def set_alt_gains(self, Kp: float, Ki: float, Kd: float):
        self.alt_pid.set_gains(Kp, Ki, Kd)

    def set_speed_gain(self, Kp: float):
        self.Kp_spd = Kp