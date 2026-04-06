# 2D Flight Simulator (Aircraft Longitudinal Dynamics + Autopilot)

A physics-based 2D flight simulator implementing longitudinal aircraft dynamics, real-time control using a PID-based autopilot, and an interactive GUI for visualization and tuning. Readme authored by Chatgpt and reviewed by Akram!

---

## System Overview

The project is structured into three tightly coupled subsystems:

### 1. Physics Engine

Implements a continuous-time aircraft model in the vertical plane using first-principles aerodynamics and numerical integration.

Defined in: 

**State vector**

* Airspeed `V` (m/s)
* Flight path angle `γ` (rad)
* Altitude `h` (m)

**Core dynamics**

- Lift:
  L = 0.5 * rho * V^2 * S * C_L

- Drag:
  D = 0.5 * rho * V^2 * S * C_D

- Lift coefficient:
  C_L = 2π * alpha

- Drag coefficient:
  C_D = C_D0 + k * alpha^2

**Equations of motion**

- dV/dt = (T*cos(alpha) - D - m*g*sin(gamma)) / m

- d(gamma)/dt = (L + T*sin(alpha) - m*g*cos(gamma)) / (m*V)

- dh/dt = V*sin(gamma)

**Numerical integration**

* 4th-order Runge-Kutta (RK4)
* Fixed timestep simulation

**Additional modeling**

* Exponential atmosphere model
* Wind disturbance
* Sensor noise injection
* Angle-of-attack saturation (±20°)

---

### 2. Control System (Autopilot)

Defined in: 

Implements a **two-loop control architecture**, mirroring real aircraft autopilot systems.

#### Outer Loop (Altitude → Pitch)

* PID controller generates pitch command:

  (\theta = K_p e + K_i \int e dt + K_d \frac{de}{dt})

* Error: (e = h_{ref} - h)

#### Inner Loop (Speed → Thrust)

* Feed-forward + proportional feedback:

  * Feed-forward compensates gravity and nominal drag
  * Feedback corrects airspeed deviation

#### Features

* Output saturation (pitch limits ±15°)
* Anti-windup integral clamping
* Tunable gains in real-time
* Separation of longitudinal dynamics and control logic

---

### 3. GUI + Visualization

Defined in: 

Built using **PyQt5 + Matplotlib**.

#### Components

**Aircraft Canvas**

* Real-time 2D side-view rendering
* Sky gradient based on altitude
* Moving ground and clouds (visual motion cues)
* Aircraft orientation based on flight-path angle
* HUD displaying:

  * Altitude
  * Speed
  * Flight path angle
  * Wind
  * Simulation time

**Live Plots**

* Altitude vs time (with reference tracking)
* Airspeed vs time
* Rolling buffer visualization

**Control Panel**

* Target altitude slider (0–5000 m)
* PID gain tuning (Kp, Ki, Kd)
* Wind disturbance input
* Sensor noise toggle
* Simulation controls:

  * Start
  * Pause / Resume
  * Reset

---

## Execution Flow

Defined in: 

1. GUI timer triggers simulation loop (20 Hz)
2. For each frame:

   * Autopilot computes `(thrust, pitch)`
   * Physics integrates state using RK4
3. State updates propagate to:

   * Animation canvas
   * Live plots
   * HUD metrics

---

## Installation

```bash
pip install numpy matplotlib PyQt5
```

---

## Run

```bash
python main.py
```

---

## Project Structure

```
2D_FlightSim/
│
├── main.py        # Entry point (Qt application bootstrap)
├── physics.py     # Aircraft dynamics + RK4 integrator
├── control.py     # PID + autopilot logic
├── Gui.py         # Full GUI, visualization, interaction
├── requirements.txt
└── README.md
```

---

## Key Engineering Characteristics

* Physically grounded model (not kinematic approximation)
* Separation of dynamics and control layers
* Deterministic integration (fixed-step RK4)
* Real-time interactive control tuning
* Disturbance injection (wind, noise)
* Control architecture aligned with real AFCS design

---

## Limitations

* 2D longitudinal model only (no lateral/yaw dynamics)
* No stall modeling beyond simple α clamp
* No actuator dynamics (instantaneous control response)
* Simplified aerodynamic coefficients

---

## Extension Directions

* 6-DOF rigid body dynamics
* Nonlinear aerodynamic lookup tables
* MPC / LQR controller replacement
* Actuator and sensor dynamics modeling
* State estimation (Kalman filtering)
* Hardware-in-the-loop integration

---

## License

MIT License
