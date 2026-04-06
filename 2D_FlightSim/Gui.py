"""
gui.py — Flight Simulator GUI

Layout
------
  Left panel  : Aircraft animation canvas (side view)
  Right panel : Live plots (altitude & velocity vs time)
  Bottom panel: Controls (sliders, gains, start/stop/reset/pause)

Uses PyQt5 + Matplotlib embedded in Qt.
"""

import sys
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QGridLayout,
    QDoubleSpinBox, QCheckBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QPolygonF
from PyQt5.QtCore import QPointF

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from physics import AircraftSim, AIRCRAFT
from control import AltitudeHoldAutopilot


# ── Constants ─────────────────────────────────────────────────────────────────
DT         = 0.05        # simulation timestep  (s)
TIMER_MS   = 50          # GUI refresh rate     (ms)  → 20 Hz
HISTORY    = 400         # number of data points to keep in plots
ALT_MIN    = 0
ALT_MAX    = 5000        # m
ALT_DEFAULT= 1000        # m
V0         = 80.0        # initial airspeed (m/s)


class AircraftCanvas(QWidget):
    """
    Custom Qt widget that draws the 2-D side-view animation.
    Renders sky gradient, terrain, clouds, and the aircraft silhouette.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 320)
        self._h       = ALT_DEFAULT   # current altitude (m)
        self._gamma   = 0.0           # flight-path angle (rad)
        self._V       = V0
        self._h_ref   = ALT_DEFAULT
        self._t       = 0.0
        self._wind    = 0.0
        self._running = False

        # Scroll offset for ground texture
        self._ground_offset = 0.0
        # Simple cloud positions [x_frac, y_frac, size]
        self._clouds = [(0.15, 0.15, 60), (0.45, 0.08, 45),
                        (0.72, 0.18, 55), (0.88, 0.10, 40)]

    def update_state(self, h, gamma, V, h_ref, t, wind=0.0, running=False):
        self._h       = h
        self._gamma   = gamma
        self._V       = V
        self._h_ref   = h_ref
        self._t       = t
        self._wind    = wind
        self._running = running
        if running:
            self._ground_offset = (self._ground_offset + V * DT * 0.3) % 120
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # ── Sky gradient ──────────────────────────────────────────────────────
        alt_frac = min(self._h / ALT_MAX, 1.0)
        sky_top  = QColor(10,  30,  80)
        sky_bot  = QColor(135, 195, 235)
        r = int(sky_top.red()   + (sky_bot.red()   - sky_top.red())   * (1 - alt_frac))
        g = int(sky_top.green() + (sky_bot.green() - sky_top.green()) * (1 - alt_frac))
        b = int(sky_top.blue()  + (sky_bot.blue()  - sky_top.blue())  * (1 - alt_frac))
        p.fillRect(0, 0, W, H, QColor(r, g, b))

        # ── Clouds ────────────────────────────────────────────────────────────
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 180)))
        cloud_offset = self._ground_offset * 0.3
        for cx, cy, sz in self._clouds:
            x = (cx * W - cloud_offset * 0.5) % W
            y = cy * H
            p.drawEllipse(int(x - sz/2), int(y - sz/3), sz, int(sz*0.6))
            p.drawEllipse(int(x - sz*0.2), int(y - sz/2), int(sz*0.7), int(sz*0.55))
            p.drawEllipse(int(x + sz*0.1), int(y - sz/3), int(sz*0.6), int(sz*0.5))

        # ── Ground ────────────────────────────────────────────────────────────
        ground_h = max(30, int(H * 0.18 * (1 - alt_frac * 0.7)))
        ground_y = H - ground_h

        # Earth fill
        p.setBrush(QBrush(QColor(60, 120, 40)))
        p.drawRect(0, ground_y, W, ground_h)

        # Runway strips / texture
        p.setBrush(QBrush(QColor(50, 100, 35)))
        for i in range(-1, W // 120 + 2):
            x = int(i * 120 - (self._ground_offset % 120))
            p.drawRect(x, ground_y, 50, ground_h)

        # Ground line
        p.setPen(QPen(QColor(30, 80, 20), 2))
        p.drawLine(0, ground_y, W, ground_y)

        # ── Altitude reference line ───────────────────────────────────────────
        h_ref_y = H - ground_h - int((self._h_ref / ALT_MAX) * (H - ground_h - 20))
        h_ref_y = max(20, min(H - ground_h - 5, h_ref_y))
        p.setPen(QPen(QColor(255, 220, 0, 150), 1, Qt.DashLine))
        p.drawLine(0, h_ref_y, W, h_ref_y)
        p.setFont(QFont("Consolas", 8))
        p.setPen(QColor(255, 220, 0, 200))
        p.drawText(5, h_ref_y - 3, f"REF {self._h_ref:.0f}m")

        # ── Aircraft ──────────────────────────────────────────────────────────
        # Map altitude to vertical position (aircraft always shown at mid-left)
        ac_x = int(W * 0.28)
        ac_y = H - ground_h - int((self._h / max(ALT_MAX, 1)) * (H - ground_h - 20))
        ac_y = max(30, min(H - ground_h - 10, ac_y))

        self._draw_aircraft(p, ac_x, ac_y, self._gamma)

        # ── HUD overlay ───────────────────────────────────────────────────────
        self._draw_hud(p, W, H)

        p.end()

    def _draw_aircraft(self, p, cx, cy, gamma):
        """Draw a stylised aircraft silhouette, rotated by gamma."""
        p.save()
        p.translate(cx, cy)
        p.rotate(-np.degrees(gamma))   # Qt y-axis is inverted

        scale = 1.8
        body_color   = QColor(220, 225, 235)
        cockpit_color= QColor(100, 180, 220)
        engine_color = QColor(80, 80, 90)
        wing_color   = QColor(190, 195, 210)

        # Fuselage
        path = QPainterPath()
        path.moveTo(-52*scale, -6*scale)
        path.lineTo( 30*scale, -6*scale)
        path.lineTo( 52*scale,  0)
        path.lineTo( 30*scale,  6*scale)
        path.lineTo(-52*scale,  6*scale)
        path.closeSubpath()
        p.setPen(QPen(QColor(100, 100, 120), 1))
        p.setBrush(QBrush(body_color))
        p.drawPath(path)

        # Cockpit
        p.setBrush(QBrush(cockpit_color))
        p.setPen(QPen(QColor(60, 140, 180), 1))
        p.drawEllipse(int(10*scale), int(-7*scale), int(22*scale), int(8*scale))

        # Main wing
        wing = QPainterPath()
        wing.moveTo(-10*scale, -6*scale)
        wing.lineTo(  5*scale, -30*scale)
        wing.lineTo( 25*scale, -30*scale)
        wing.lineTo( 20*scale, -6*scale)
        wing.closeSubpath()
        p.setBrush(QBrush(wing_color))
        p.setPen(QPen(QColor(100,100,120), 1))
        p.drawPath(wing)

        # Horizontal stabiliser (tail)
        htail = QPainterPath()
        htail.moveTo(-40*scale, -6*scale)
        htail.lineTo(-48*scale, -18*scale)
        htail.lineTo(-38*scale, -18*scale)
        htail.lineTo(-30*scale, -6*scale)
        htail.closeSubpath()
        p.drawPath(htail)

        # Vertical stabiliser
        vtail = QPainterPath()
        vtail.moveTo(-45*scale, -6*scale)
        vtail.lineTo(-50*scale, -22*scale)
        vtail.lineTo(-38*scale, -6*scale)
        vtail.closeSubpath()
        p.drawPath(vtail)

        # Engine pod
        p.setBrush(QBrush(engine_color))
        p.setPen(QPen(QColor(50,50,60), 1))
        p.drawEllipse(int(-5*scale), int(-38*scale), int(18*scale), int(10*scale))

        p.restore()

    def _draw_hud(self, p, W, H):
        """Heads-up display: speed, altitude, FPA."""
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        margin = 10
        line_h = 16
        texts = [
            (f"ALT  {self._h:7.1f} m",   QColor(100, 255, 100)),
            (f"SPD  {self._V:7.1f} m/s", QColor(100, 200, 255)),
            (f"FPA  {np.degrees(self._gamma):+6.1f}°",  QColor(255, 200, 100)),
            (f"WIND {self._wind:+6.1f} m/s", QColor(200, 150, 255)),
            (f"T    {self._t:7.1f} s",   QColor(180, 180, 180)),
        ]
        box_w, box_h = 170, len(texts) * line_h + 10
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(margin, margin, box_w, box_h, 6, 6)

        for i, (txt, col) in enumerate(texts):
            p.setPen(col)
            p.drawText(margin + 8, margin + 14 + i * line_h, txt)


class PlotCanvas(FigureCanvas):
    """Matplotlib figure embedded in Qt, with two live plots."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor='#0d1117')
        super().__init__(self.fig)
        self.setParent(parent)

        gs = gridspec.GridSpec(2, 1, figure=self.fig,
                               hspace=0.45, top=0.93, bottom=0.1,
                               left=0.14, right=0.96)

        style = dict(facecolor='#0d1a2a')
        self.ax_alt = self.fig.add_subplot(gs[0], **style)
        self.ax_vel = self.fig.add_subplot(gs[1], **style)

        for ax in (self.ax_alt, self.ax_vel):
            ax.tick_params(colors='#8899aa', labelsize=8)
            ax.xaxis.label.set_color('#8899aa')
            ax.yaxis.label.set_color('#8899aa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#223344')

        self.ax_alt.set_title("Altitude", color='#aaccee', fontsize=9, pad=4)
        self.ax_alt.set_ylabel("h (m)",   color='#aaccee', fontsize=8)
        self.ax_alt.set_xlabel("t (s)",   color='#8899aa', fontsize=8)

        self.ax_vel.set_title("Airspeed", color='#aaccee', fontsize=9, pad=4)
        self.ax_vel.set_ylabel("V (m/s)", color='#aaccee', fontsize=8)
        self.ax_vel.set_xlabel("t (s)",   color='#8899aa', fontsize=8)

        self.line_alt,  = self.ax_alt.plot([], [], color='#44dd88', lw=1.5, label='h')
        self.line_href, = self.ax_alt.plot([], [], color='#ffdd00', lw=1,
                                           linestyle='--', label='h_ref')
        self.line_vel,  = self.ax_vel.plot([], [], color='#44aaff', lw=1.5)

        self.ax_alt.legend(fontsize=7, facecolor='#0d1a2a',
                           labelcolor='white', loc='upper right')

        self._t    = deque(maxlen=HISTORY)
        self._h    = deque(maxlen=HISTORY)
        self._href = deque(maxlen=HISTORY)
        self._V    = deque(maxlen=HISTORY)

    def append(self, t, h, h_ref, V):
        self._t.append(t)
        self._h.append(h)
        self._href.append(h_ref)
        self._V.append(V)

    def refresh(self):
        if len(self._t) < 2:
            return
        ta = list(self._t)
        ha = list(self._h)
        hr = list(self._href)
        Va = list(self._V)

        self.line_alt.set_data(ta, ha)
        self.line_href.set_data(ta, hr)
        self.line_vel.set_data(ta, Va)

        for ax in (self.ax_alt, self.ax_vel):
            ax.relim()
            ax.autoscale_view()

        self.draw_idle()


class FlightSimWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Aircraft Flight Simulator")
        self.setMinimumSize(1150, 680)

        # ── Simulation objects ────────────────────────────────────────────────
        self.sim = AircraftSim(V0=V0, gamma0=0.0, h0=ALT_DEFAULT, dt=DT)
        self.ap  = AltitudeHoldAutopilot(
            Kp_alt=0.025, Ki_alt=0.0008, Kd_alt=0.20,
            Kp_spd=250.0,
            T_min=AIRCRAFT['T_min'], T_max=AIRCRAFT['T_max'],
            V_ref=V0, m=AIRCRAFT['m']
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._running = False
        self._paused  = False
        self._h_ref   = float(ALT_DEFAULT)

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()

        # ── Timer ─────────────────────────────────────────────────────────────
        self.timer = QTimer()
        self.timer.setInterval(TIMER_MS)
        self.timer.timeout.connect(self._tick)

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Title
        title = QLabel("✈  2D Longitudinal Flight Simulator")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#88ccff;"
                            "padding:4px 0;")
        root.addWidget(title)

        # Main splitter: canvas | plots
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Left: animation
        self.canvas = AircraftCanvas()
        self.canvas.setStyleSheet("background:#0d1117; border-radius:6px;")
        splitter.addWidget(self.canvas)

        # Right: plots
        self.plots = PlotCanvas()
        splitter.addWidget(self.plots)
        splitter.setSizes([550, 450])

        # Bottom controls
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background:#0d1a2a; border-radius:6px;")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(14)
        root.addWidget(ctrl_frame)

        ctrl_layout.addWidget(self._build_sim_controls())
        ctrl_layout.addWidget(self._build_altitude_control())
        ctrl_layout.addWidget(self._build_pid_gains())
        ctrl_layout.addWidget(self._build_environment())
        ctrl_layout.addStretch()

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0a0f1a; color: #ccd9e8; }
            QGroupBox { border: 1px solid #223344; border-radius:5px;
                        margin-top:10px; font-weight:bold; color:#88aacc; font-size:10px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; top:0px; }
            QLabel  { color:#aabbcc; font-size:10px; }
            QSlider::groove:horizontal { height:4px; background:#223344; border-radius:2px; }
            QSlider::handle:horizontal { width:14px; height:14px; margin:-5px 0;
                                         background:#4488cc; border-radius:7px; }
            QSlider::sub-page:horizontal { background:#336699; border-radius:2px; }
            QPushButton { background:#1a3355; color:#88ccff; border:1px solid #336699;
                          border-radius:5px; padding:5px 14px; font-size:11px; }
            QPushButton:hover   { background:#2244aa; }
            QPushButton:pressed { background:#112233; }
            QPushButton:disabled{ background:#111822; color:#445566; }
            QDoubleSpinBox { background:#0d1a2a; color:#aaccee; border:1px solid #223344;
                             border-radius:4px; padding:2px; }
            QCheckBox { color:#aabbcc; font-size:10px; }
        """)

    def _build_sim_controls(self):
        grp = QGroupBox("Simulation")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        self.btn_start = QPushButton("▶  Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_pause.setEnabled(False)
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.clicked.connect(self._on_reset)

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_pause)
        lay.addWidget(self.btn_reset)
        return grp

    def _build_altitude_control(self):
        grp = QGroupBox("Target Altitude")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        self.lbl_alt = QLabel(f"h_ref = {self._h_ref:.0f} m")
        self.lbl_alt.setStyleSheet("color:#ffdd44; font-weight:bold;")
        lay.addWidget(self.lbl_alt)

        self.sld_alt = QSlider(Qt.Horizontal)
        self.sld_alt.setRange(ALT_MIN, ALT_MAX)
        self.sld_alt.setValue(int(self._h_ref))
        self.sld_alt.setFixedWidth(180)
        self.sld_alt.valueChanged.connect(self._on_alt_changed)
        lay.addWidget(self.sld_alt)

        lay.addWidget(QLabel("0 m ←——→ 5000 m"))
        return grp

    def _build_pid_gains(self):
        grp = QGroupBox("PID Gains (altitude loop)")
        lay = QGridLayout(grp)
        lay.setSpacing(4)

        def make_spinbox(val, lo, hi, step):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(4)
            sb.setValue(val)
            sb.setFixedWidth(85)
            return sb

        self.sp_kp = make_spinbox(0.025, 0.0, 2.0,  0.005)
        self.sp_ki = make_spinbox(0.0008, 0.0, 0.5, 0.0001)
        self.sp_kd = make_spinbox(0.20,  0.0, 5.0,  0.05)

        lay.addWidget(QLabel("Kp"), 0, 0); lay.addWidget(self.sp_kp, 0, 1)
        lay.addWidget(QLabel("Ki"), 1, 0); lay.addWidget(self.sp_ki, 1, 1)
        lay.addWidget(QLabel("Kd"), 2, 0); lay.addWidget(self.sp_kd, 2, 1)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_gains_changed)
        lay.addWidget(apply_btn, 3, 0, 1, 2)
        return grp

    def _build_environment(self):
        grp = QGroupBox("Environment")
        lay = QGridLayout(grp)
        lay.setSpacing(4)

        # Wind
        lay.addWidget(QLabel("Wind (m/s)"), 0, 0)
        self.sp_wind = QDoubleSpinBox()
        self.sp_wind.setRange(-30, 30)
        self.sp_wind.setValue(0.0)
        self.sp_wind.setSingleStep(1.0)
        self.sp_wind.setFixedWidth(70)
        self.sp_wind.valueChanged.connect(self._on_wind_changed)
        lay.addWidget(self.sp_wind, 0, 1)

        # Noise
        self.chk_noise = QCheckBox("Sensor noise")
        self.chk_noise.stateChanged.connect(self._on_noise_changed)
        lay.addWidget(self.chk_noise, 1, 0, 1, 2)

        # Status label
        self.lbl_status = QLabel("Status: Stopped")
        self.lbl_status.setStyleSheet("color:#66ffaa; font-size:9px;")
        lay.addWidget(self.lbl_status, 2, 0, 1, 2)
        return grp

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation loop
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self):
        """Called by QTimer every TIMER_MS ms. Advances physics + updates GUI."""
        if not self._running or self._paused:
            return

        # Step physics (multiple physics steps per GUI frame for accuracy)
        steps_per_frame = max(1, int(TIMER_MS / 1000 / DT))
        for _ in range(steps_per_frame):
            # Autopilot computes commands
            thrust, theta = self.ap.compute(
                h     = self.sim.h,
                h_ref = self._h_ref,
                V     = self.sim.V,
                gamma = self.sim.gamma,
                dt    = DT,
            )
            self.sim.thrust = thrust
            self.sim.theta  = theta
            self.sim.step()

        # Update plots data
        self.plots.append(self.sim.t, self.sim.h, self._h_ref, self.sim.V)

        # Refresh widgets (every frame)
        self.canvas.update_state(
            h=self.sim.h, gamma=self.sim.gamma, V=self.sim.V,
            h_ref=self._h_ref, t=self.sim.t,
            wind=self.sim.wind_speed, running=True
        )
        self.plots.refresh()

        # Status label
        err = self._h_ref - self.sim.h
        self.lbl_status.setText(
            f"t={self.sim.t:.1f}s  Δh={err:+.1f}m  T={self.sim.thrust/1000:.1f}kN"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Slot handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_start(self):
        if not self._running:
            self._running = True
            self._paused  = False
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.lbl_status.setText("Status: Running")
            self.timer.start()

    def _on_pause(self):
        self._paused = not self._paused
        self.btn_pause.setText("▶  Resume" if self._paused else "⏸  Pause")
        self.lbl_status.setText("Status: Paused" if self._paused else "Status: Running")

    def _on_reset(self):
        self.timer.stop()
        self._running = False
        self._paused  = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")

        # Reset sim and autopilot
        self.sim = AircraftSim(V0=V0, gamma0=0.0, h0=self._h_ref, dt=DT)
        self.sim.wind_speed = self.sp_wind.value()
        self.sim.noise_std  = 1.0 if self.chk_noise.isChecked() else 0.0
        self.ap.reset()

        # Clear plots
        self.plots._t.clear(); self.plots._h.clear()
        self.plots._href.clear(); self.plots._V.clear()
        self.plots.refresh()

        self.canvas.update_state(
            self.sim.h, self.sim.gamma, self.sim.V,
            self._h_ref, self.sim.t, running=False
        )
        self.lbl_status.setText("Status: Stopped")

    def _on_alt_changed(self, val):
        self._h_ref = float(val)
        self.lbl_alt.setText(f"h_ref = {self._h_ref:.0f} m")

    def _on_gains_changed(self):
        self.ap.set_alt_gains(
            Kp=self.sp_kp.value(),
            Ki=self.sp_ki.value(),
            Kd=self.sp_kd.value(),
        )
        self.ap.alt_pid.reset()   # avoid integral windup on gain change

    def _on_wind_changed(self, val):
        self.sim.wind_speed = val

    def _on_noise_changed(self, state):
        self.sim.noise_std = 1.0 if state == Qt.Checked else 0.0