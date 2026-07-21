"""
=============================================================================
 interactive_rsf_spring_slider.py   --   v0.2   (Stage 3)
=============================================================================
 PhET-style interactive rate-and-state spring-slider.

 Run:  python interactive_rsf_spring_slider.py
 Then open  http://127.0.0.1:8080  in a browser.

-----------------------------------------------------------------------------
 WHAT THIS FILE IS ALLOWED TO DO
-----------------------------------------------------------------------------
 Draw things and collect clicks. That is all.

 It contains NO physics. It never touches lnV, theta, delta or solve_ivp. It
 cannot move the block: the block's position on screen is computed from the
 numerical solution, and the only thing a mouse drag can do is impose a
 load-point displacement, which is handed to the controller and integrated by
 rsf_engine. If the picture ever looks wrong, the fix belongs in the engine or
 the controller, never here.

 Layering:      rsf_engine.py  ->  simulation_controller.py  ->  this file
                (physics)          (operation)                  (pixels)

 Every displayed number originates in rsf_engine.derived(), reaches this file
 through SimulationController.snapshot() / plot_data(), and is only formatted
 here. The animation, the live values and the graphs therefore cannot disagree
 with one another: they are three views of one array.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from nicegui import app, ui

import rsf_engine as E
import simulation_controller as SC
from simulation_controller import (LoadingMode, OpState, SimulationController,
                                   SimulationWorker, THRESHOLDS)

# =============================================================================
# SCENE GEOMETRY (pure presentation constants, in SVG user units = px)
# =============================================================================

W, H = 780, 300
Y_SURFACE = 214
BLOCK_W, BLOCK_H = 118, 66
Y_BLOCK = Y_SURFACE - BLOCK_H
X_HANDLE = 648                 # the load-point handle is drawn at a FIXED x
HANDLE_W, HANDLE_H = 24, 84
SPRING_REST_PX = 150
SPRING_MIN_PX, SPRING_MAX_PX = 46, 300
HATCH_PERIOD = 34

# Colours
C_ROCK = "#6b7a86"
C_ROCK_D = "#4a5761"
C_BLOCK = "#3b6ea5"
C_BLOCK_E = "#e8613c"
C_SPRING = "#2f6f4f"
C_HANDLE = "#8a5a2b"
C_TXT = "#1d2429"
C_MUTE = "#7b8794"
C_VLP = "#8a5a2b"
C_V = "#3b6ea5"


def _fmt(x: float, unit: str = "", sig: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return f"0 {unit}".strip()
    a = abs(x)
    if 1e-3 <= a < 1e5:
        s = f"{x:,.{sig}g}"
    else:
        s = f"{x:.{sig}e}"
    return f"{s} {unit}".strip()


def _arrow(x: float, y: float, length: float, colour: str,
           label: str) -> str:
    """Right-pointing arrow of the given pixel length."""
    length = float(np.clip(length, 0.0, 190.0))
    if length < 3:
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{colour}"/>'
                f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" font-size="11" '
                f'fill="{colour}">{label}</text>')
    x2 = x + length
    return (f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2 - 9:.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="3"/>'
            f'<polygon points="{x2:.1f},{y:.1f} {x2 - 10:.1f},{y - 5:.1f} '
            f'{x2 - 10:.1f},{y + 5:.1f}" fill="{colour}"/>'
            f'<text x="{x:.1f}" y="{y - 8:.1f}" font-size="11" '
            f'fill="{colour}">{label}</text>')


def _spring_path(x0: float, x1: float, y: float, coils: int,
                 amp: float) -> str:
    """Zigzag from x0 to x1. The number of coils is fixed; only the pitch
    changes, so stretching looks like stretching."""
    n = max(coils * 2, 4)
    pts = [(x0, y)]
    span = x1 - x0
    lead = min(16.0, abs(span) * 0.12)
    for i in range(n + 1):
        f = i / n
        xx = x0 + lead + f * (span - 2 * lead)
        yy = y + (amp if i % 2 == 0 else -amp)
        pts.append((xx, yy))
    pts.append((x1, y))
    return " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)


def build_scene_svg(s: SC.Snapshot, mag: float, flash: bool,
                    shake: float, params: E.RSFParams) -> str:
    """Build the whole scene. Positions come ONLY from u_lp, delta, e0 and e.

    Camera convention: the handle is drawn at a fixed x and the ground pattern
    scrolls, so the load point's absolute advance is visible as ground motion
    while the block's position relative to the handle shows the spring state.
    Block screen offset = -mag * (e - e0) = mag * (delta - u_lp), i.e. it is
    derived, not animated independently.
    """
    e_rel = s.e - s.e0                     # = u_lp - delta  [m]
    spring_px = float(np.clip(SPRING_REST_PX + mag * e_rel,
                              SPRING_MIN_PX, SPRING_MAX_PX))
    bx = X_HANDLE - spring_px - BLOCK_W
    sh = shake

    scroll = (mag * s.u_lp) % HATCH_PERIOD
    parts: List[str] = []
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fbfcfd"/>')

    # ---- frictional surface / rock -------------------------------------
    parts.append(f'<rect x="0" y="{Y_SURFACE}" width="{W}" '
                 f'height="{H - Y_SURFACE}" fill="{C_ROCK}"/>')
    x = -HATCH_PERIOD - scroll
    while x < W + HATCH_PERIOD:
        parts.append(f'<line x1="{x:.1f}" y1="{H}" x2="{x + 26:.1f}" '
                     f'y2="{Y_SURFACE}" stroke="{C_ROCK_D}" '
                     f'stroke-width="2" opacity="0.55"/>')
        x += HATCH_PERIOD
    # roughness teeth on the interface
    tx = -12 - (scroll * 0.5) % 12
    teeth = []
    while tx < W + 12:
        teeth.append(f"{tx:.1f},{Y_SURFACE} {tx + 6:.1f},{Y_SURFACE - 4} "
                     f"{tx + 12:.1f},{Y_SURFACE}")
        tx += 12
    parts.append(f'<polyline points="{" ".join(teeth)}" fill="none" '
                 f'stroke="#2b343b" stroke-width="1.6"/>')
    parts.append(f'<text x="10" y="{H - 10}" font-size="11" fill="#e8eef3">'
                 f'frictional interface &#183; effective normal stress '
                 f'&#963;n = {params.sigma_n/1e6:.3g} MPa</text>')

    # ---- block ----------------------------------------------------------
    bcol = C_BLOCK_E if flash else C_BLOCK
    parts.append(f'<rect x="{bx + sh:.1f}" y="{Y_BLOCK}" width="{BLOCK_W}" '
                 f'height="{BLOCK_H}" rx="4" fill="{bcol}" '
                 f'stroke="#1d3f63" stroke-width="2"/>')
    parts.append(f'<text x="{bx + BLOCK_W/2 + sh:.1f}" y="{Y_BLOCK + 30:.1f}" '
                 f'font-size="12" fill="#eaf2fa" text-anchor="middle">'
                 f'SLIDER BLOCK</text>')
    parts.append(f'<text x="{bx + BLOCK_W/2 + sh:.1f}" y="{Y_BLOCK + 47:.1f}" '
                 f'font-size="11" fill="#c4dcf2" text-anchor="middle">'
                 f'&#956; = {s.mu:.4f}</text>')

    # ---- spring ----------------------------------------------------------
    # Tension is encoded by BOTH colour and thickness.
    tau_frac = float(np.clip(s.tau_total / max(params.sigma_n * 0.8, 1), 0, 1))
    sw = 2.2 + 3.4 * tau_frac
    r = int(47 + 170 * tau_frac)
    g = int(111 - 40 * tau_frac)
    b = int(79 - 20 * tau_frac)
    scol = f"rgb({r},{max(g,0)},{max(b,0)})"
    y_spring = Y_BLOCK + BLOCK_H / 2
    parts.append(f'<polyline points="'
                 f'{_spring_path(bx + BLOCK_W + sh, X_HANDLE, y_spring, 9, 15)}"'
                 f' fill="none" stroke="{scol}" stroke-width="{sw:.2f}" '
                 f'stroke-linejoin="round"/>')
    parts.append(f'<text x="{(bx + BLOCK_W + X_HANDLE)/2:.1f}" '
                 f'y="{y_spring + 40:.1f}" font-size="11" fill="{C_MUTE}" '
                 f'text-anchor="middle">spring k = {params.k:.3g} Pa/m</text>')

    # ---- load-point handle ----------------------------------------------
    parts.append(f'<rect x="{X_HANDLE:.1f}" y="{y_spring - HANDLE_H/2:.1f}" '
                 f'width="{HANDLE_W}" height="{HANDLE_H}" rx="3" '
                 f'fill="{C_HANDLE}" stroke="#5c3c1c" stroke-width="2"/>')
    parts.append(f'<line x1="{X_HANDLE + HANDLE_W:.1f}" y1="{y_spring:.1f}" '
                 f'x2="{W}" y2="{y_spring:.1f}" stroke="{C_HANDLE}" '
                 f'stroke-width="7"/>')
    parts.append(f'<text x="{X_HANDLE + 12:.1f}" y="{y_spring - HANDLE_H/2 - 8:.1f}"'
                 f' font-size="11" fill="{C_HANDLE}" text-anchor="middle">'
                 f'LOAD POINT</text>')

    # ---- velocity arrows (logarithmic length: V spans ~10 decades) -------
    def dec_len(v: float) -> float:
        if v <= 0:
            return 0.0
        return float(np.clip(22.0 * (math.log10(v) + 11.0), 0.0, 180.0))

    parts.append(_arrow(X_HANDLE - 4, y_spring - HANDLE_H / 2 - 26,
                        dec_len(s.V_lp), C_VLP,
                        f"V_lp = {_fmt(s.V_lp, 'm/s', 3)}"))
    parts.append(_arrow(bx + sh, Y_BLOCK - 30, dec_len(s.V), C_V,
                        f"V = {_fmt(s.V, 'm/s', 3)}"))
    parts.append(f'<text x="{W - 8}" y="{H - 10}" font-size="10" '
                 f'fill="#e8eef3" text-anchor="end">arrow length is '
                 f'LOGARITHMIC in velocity</text>')

    # ---- stress indicators ----------------------------------------------
    bx0, by0, bw, bh = 14, 20, 210, 12
    smax = max(s.tau_total, s.tau_friction, 1.0) * 1.25
    parts.append(f'<text x="{bx0}" y="{by0 - 6}" font-size="11" '
                 f'fill="{C_TXT}">driving stress &#964;_total = '
                 f'{s.tau_total/1e6:.4f} MPa</text>')
    parts.append(f'<rect x="{bx0}" y="{by0}" width="{bw}" height="{bh}" '
                 f'fill="#e6eaee" rx="2"/>')
    parts.append(f'<rect x="{bx0}" y="{by0}" '
                 f'width="{bw * s.tau_total / smax:.1f}" height="{bh}" '
                 f'fill="#c2603a" rx="2"/>')
    parts.append(f'<text x="{bx0}" y="{by0 + bh + 18}" font-size="11" '
                 f'fill="{C_TXT}">frictional resistance &#964;_friction = '
                 f'{s.tau_friction/1e6:.4f} MPa</text>')
    parts.append(f'<rect x="{bx0}" y="{by0 + bh + 24}" width="{bw}" '
                 f'height="{bh}" fill="#e6eaee" rx="2"/>')
    parts.append(f'<rect x="{bx0}" y="{by0 + bh + 24}" '
                 f'width="{bw * s.tau_friction / smax:.1f}" height="{bh}" '
                 f'fill="#2f6f4f" rx="2"/>')

    # ---- operational state + magnification note --------------------------
    badge = "#c2603a" if flash else "#33414c"
    parts.append(f'<rect x="{W - 268}" y="14" width="254" height="30" rx="5" '
                 f'fill="{badge}"/>')
    parts.append(f'<text x="{W - 141}" y="34" font-size="14" fill="#ffffff" '
                 f'text-anchor="middle">{s.visual_state_label}</text>')
    parts.append(f'<text x="{W - 141}" y="58" font-size="10.5" '
                 f'fill="{C_MUTE}" text-anchor="middle">'
                 f'Visual displacement exaggerated by &#215;'
                 f'{mag:,.0f}</text>')

    if flash:
        parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" '
                     f'fill="#e8613c" opacity="0.10"/>')

    return "".join(parts)


# =============================================================================
# LIVE-VALUE METADATA  (meaning, unit, and PROVENANCE of every quantity)
# =============================================================================
# provenance: imposed | integrated | calculated | model-defined | numerical
EXPLORE_FIELDS: List[Tuple[str, str, str, str, str]] = [
    ("t", "Model time", "s", "integrated",
     "Time since Reset, in the model's own clock. Not wall-clock time: the "
     "playback speed sets how fast you watch it pass."),
    ("V_lp", "Load-point velocity V_lp", "m/s", "imposed",
     "The velocity at which the machine drives the far end of the spring. "
     "You impose this; it is a boundary condition, not a response."),
    ("V", "Block slip velocity V", "m/s", "integrated",
     "How fast the block itself is sliding. This is a RESPONSE of the "
     "system, obtained by integrating the equations. It is not V_lp and it "
     "is not a rupture-front velocity."),
    ("e", "Spring extension e", "m", "calculated",
     "Total extension e = e0 + u_lp - delta, where e0 is the initial "
     "pre-extension frozen at Reset. Multiply by k to get the driving "
     "stress."),
    ("tau_total", "Driving stress tau_total", "Pa", "calculated",
     "tau_total = sigma_n*mu + eta*V, and equals k*e. This is the total "
     "stress driving the block, NOT the frictional strength alone."),
    ("mu", "Friction coefficient mu", "-", "calculated",
     "mu = mu0 + a ln(V/V0) + b ln(V0 theta/Dc). Dimensionless."),
    ("theta", "State variable theta", "s", "integrated",
     "The evolving condition of the interface, with units of time. NOT "
     "universally 'contact age': that reading belongs to the ageing law and "
     "is not equally literal for every evolution law."),
    ("V_theta_over_Dc", "V*theta/Dc", "-", "calculated",
     "The dimensionless state group. Exactly 1 at steady state."),
]


# =============================================================================
# THE APPLICATION
# =============================================================================

class SpringSliderUI:

    def __init__(self) -> None:
        self.c = SimulationController("unstable", V_lp=1e-6)
        self.worker = SimulationWorker(self.c, hz=self.c.SCENE_HZ)
        self.mag = 2.0e5                # visual magnification, smoothed
        self._e_range = 1e-5
        self._shake_frames = 0
        self.manual_mode = False
        self._plot_dirty = True
        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:
        ui.add_head_html(
            "<style>body{background:#f4f6f8} .v-num{font-family:"
            "ui-monospace,SFMono-Regular,Menlo,monospace}</style>")

        with ui.header().classes("items-center justify-between px-4 py-2"):
            ui.label("Rate-and-State Spring-Slider  --  interactive "
                     "0-D quasi-dynamic simulator").classes("text-lg")
            with ui.row().classes("items-center gap-4"):
                self.lbl_time = ui.label("t = 0 s").classes("v-num")
                self.lbl_state = ui.badge("--").props("color=blue-grey-8")

        self.err_banner = ui.card().classes(
            "w-full bg-red-2 border-l-4 border-red-8").style("display:none")
        with self.err_banner:
            ui.label("Integration failed").classes("text-red-10 text-bold")
            self.err_text = ui.label("").classes("text-xs whitespace-pre-wrap")
            ui.button("Reset", on_click=self.on_reset).props("color=red")

        with ui.tabs().classes("w-full") as tabs:
            t_ex = ui.tab("Explore")
            t_gr = ui.tab("Graphs")
            t_ph = ui.tab("Physics")
        with ui.tab_panels(tabs, value=t_ex).classes("w-full"):
            with ui.tab_panel(t_ex):
                self.build_explore()
            with ui.tab_panel(t_gr):
                self.build_graphs()
            with ui.tab_panel(t_ph):
                self.build_physics()

        self.build_controls()

        ui.timer(1.0 / self.c.SCENE_HZ, self.tick_scene)
        ui.timer(1.0 / self.c.PLOT_HZ, self.tick_plots)
        self.worker.start()
        app.on_shutdown(self.worker.stop)

    # ------------------------------------------------------------------
    def build_explore(self) -> None:
        with ui.row().classes("w-full items-start gap-4 no-wrap"):
            with ui.card().classes("p-2"):
                self.scene = ui.interactive_image(
                    size=(W, H), content="", cross=False, sanitize=False,
                    on_mouse=self.on_mouse,
                    events=["mousedown", "mousemove", "mouseup",
                            "mouseleave"]).classes("w-[780px]")
                self.lbl_mag = ui.label("").classes("text-xs text-grey-8")
                self.lbl_manual = ui.label("").classes("text-xs text-orange-9")

            with ui.card().classes("min-w-[330px]"):
                ui.label("Live values").classes("text-bold")
                ui.label("hover any row for meaning, unit and provenance"
                         ).classes("text-xs text-grey-7")
                self.value_labels: Dict[str, ui.label] = {}
                for key, name, unit, prov, meaning in EXPLORE_FIELDS:
                    with ui.row().classes(
                            "w-full justify-between items-center py-0.5"):
                        ui.label(name).classes("text-xs")
                        self.value_labels[key] = ui.label("--").classes(
                            "text-xs v-num text-right")
                        ui.tooltip(f"{meaning}\n\nSI unit: {unit}\n"
                                   f"This quantity is {prov.upper()}.")
                ui.separator()
                with ui.row().classes("w-full justify-between"):
                    ui.label("k / kc_QD").classes("text-xs")
                    self.lbl_kkc = ui.label("--").classes("text-xs v-num")
                    ui.tooltip(
                        "Stiffness ratio against the QUASI-DYNAMIC critical "
                        "stiffness of the system actually implemented here, "
                        "kc_QD = [sigma_n(b-a) - eta*V_ss]/Dc. Below 1 the "
                        "steady state is unstable. Defined only for constant "
                        "positive V_lp. MODEL-DEFINED.")
                with ui.row().classes("w-full justify-between"):
                    ui.label("Operational state").classes("text-xs")
                    self.lbl_op = ui.label("--").classes("text-xs")
                    ui.tooltip(
                        "An INTERFACE CONVENTION for reading the animation, "
                        "not a universal physical phase. The exact rule that "
                        "produced it is shown in the Physics tab.")

    # ------------------------------------------------------------------
    def build_graphs(self) -> None:
        ui.label("Graphs are drawn from the same stored History as the "
                 "animation, decimated for display only. The full history is "
                 "retained. Dashed vertical lines mark event onsets."
                 ).classes("text-xs text-grey-8")
        self.ch_v = ui.echart(self._chart_opts(
            "V_lp and V vs time  (V on a logarithmic axis)", log=True))\
            .classes("w-full h-64")
        self.ch_tau = ui.echart(self._chart_opts(
            "tau_total and tau_friction vs time  [MPa]")).classes("w-full h-64")
        self.ch_disp = ui.echart(self._chart_opts(
            "u_lp (load point) and delta (block slip) vs time  [um]"))\
            .classes("w-full h-64")

    @staticmethod
    def _chart_opts(title: str, log: bool = False) -> dict:
        return {
            "title": {"text": title, "textStyle": {"fontSize": 13}},
            "tooltip": {"trigger": "axis"},
            "legend": {"top": 24},
            "grid": {"left": 62, "right": 24, "top": 58, "bottom": 34},
            "xAxis": {"type": "value", "name": "t [s]", "scale": True},
            "yAxis": {"type": "log" if log else "value", "scale": True},
            "animation": False,
            "series": [],
        }

    # ------------------------------------------------------------------
    def build_physics(self) -> None:
        ui.markdown("""
### Governing equations (all SI)

**Constitutive law** &nbsp; `mu(V,theta) = mu0 + a ln(V/V0) + b ln(V0 theta/Dc)`
&nbsp; [model-dependent]

**State evolution** &nbsp; ageing: `dtheta/dt = 1 - V theta/Dc` &nbsp;&nbsp;
slip: `dtheta/dt = -(V theta/Dc) ln(V theta/Dc)`

**Spring** &nbsp; `dtau/dt = k (V_lp - V)` &nbsp; [exact within the 0-D model]

**Quasi-dynamic balance** &nbsp; `tau = sigma_n mu + eta V`, &nbsp;
`eta = G/(2 cs)` &nbsp; [approximation]

**Integrated form** &nbsp;
`d(lnV)/dt = [k(V_lp - V) - sigma_n b (dtheta/dt)/theta] / (sigma_n a + eta V)`

**Spring bookkeeping** &nbsp; `e = e0 + u_lp - delta`, &nbsp; `tau_spring = k e`,
&nbsp; residual `= tau_spring - tau_total` is **analytically zero**, so any
non-zero value is integration error.
""").classes("text-sm")

        with ui.row().classes("w-full gap-4 items-start no-wrap"):
            with ui.card().classes("flex-1"):
                ui.label("Parameters and stiffness").classes("text-bold")
                self.phys_params = ui.label("").classes(
                    "text-xs v-num whitespace-pre-wrap")
            with ui.card().classes("flex-1"):
                ui.label("Numerical diagnostics").classes("text-bold")
                self.phys_num = ui.label("").classes(
                    "text-xs v-num whitespace-pre-wrap")

        with ui.card().classes("w-full"):
            ui.label("Operational state: physical vs visual").classes(
                "text-bold")
            ui.label(
                "The PHYSICAL label is computed from the stored history "
                "sample and is what the scientific record contains. The "
                "VISUAL label may differ for up to "
                f"{self.c.EVENT_FLASH_HOLD_S:.2f} s of WALL-CLOCK time after "
                "a rapid-slip event, because an event lasting ~0.03 s of "
                "model time need not coincide with a screen refresh and "
                "would otherwise be invisible. The latch changes nothing "
                "else: not the trajectory, not the event duration, not the "
                "catalogue, and it never claims V is above threshold."
            ).classes("text-xs")
            self.phys_state = ui.label("").classes(
                "text-xs v-num whitespace-pre-wrap")

        with ui.card().classes("w-full bg-amber-1"):
            ui.label("MODEL LIMITATIONS  --  these never go away").classes(
                "text-bold text-amber-10")
            ui.markdown("""
* **Zero-dimensional.** One block, one slip value, one velocity.
* **Quasi-dynamic.** Radiation damping only.
* **No nucleation length.** `Lc` cannot be computed here; it is a property of
  a spatially extended fault.
* **No rupture front.** The "nucleation" you see is a *temporal* acceleration,
  not a spatial process. There is no `Vr`.
* **No spatial fault boundaries**, no edge effects, no sample geometry.
* **No acoustic emissions** and no AE locations.
* **Radiation damping is not full inertia.** It omits wave travel time,
  interference, reflected phases and complete inertial stress redistribution.
  It cannot produce a seismogram.
* **Successful execution is not scientific validation.** That this program
  runs, and that its test suites pass, shows only that it solves the
  equations it claims to solve.
* The **1 mm/s event threshold is an operational convention**, not physics.
  The attribution of that value to Romanet et al. (2018) is flagged in the
  project text as UNVERIFIED.
""").classes("text-xs")

    # ------------------------------------------------------------------
    def build_controls(self) -> None:
        with ui.footer().classes("bg-grey-2 p-2"):
            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("items-center gap-2 w-full"):
                    self.btn_play = ui.button(
                        "Play", on_click=self.on_play).props("color=primary")
                    ui.button("Pause", on_click=self.on_pause).props("outline")
                    ui.button(f"Single Step ({self.c.SINGLE_STEP_DT:g} s)",
                              on_click=self.on_step).props("outline")
                    ui.button("Reset", on_click=self.on_reset).props("outline")
                    ui.button("Hold (V_lp = 0)",
                              on_click=self.on_hold).props("outline color=orange")
                    ui.separator().props("vertical")
                    ui.label("Playback").classes("text-xs")
                    ui.toggle(list(self.c.SPEEDS), value="Normal",
                              on_change=lambda e: self.c.set_speed(e.value)
                              ).props("dense")
                    ui.tooltip("PLAYBACK speed, in model seconds per real "
                               "second. A display setting: it does not change "
                               "V_lp or any physics.")
                    ui.switch("Adaptive advancement", value=True,
                              on_change=lambda e: setattr(
                                  self.c, "adaptive_advance", e.value)
                              ).props("dense")
                    ui.tooltip("Caps the model time per frame so a ~0.03 s "
                               "event is not stepped over. Changes the chunk "
                               "size, not the equations.")

                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Loading mode").classes("text-xs")
                    ui.toggle({False: "Velocity control", True: "Manual pull"},
                              value=False,
                              on_change=self.on_mode).props("dense")
                    ui.separator().props("vertical")
                    self.lbl_vlp = ui.label("").classes("text-xs v-num")
                    self.sld_vlp = ui.slider(
                        min=math.log10(SC.V_LP_MIN),
                        max=math.log10(SC.V_LP_MAX),
                        step=0.05, value=-6.0,
                        on_change=self.on_vlp).props("dense label")\
                        .classes("w-64")
                    ui.tooltip(
                        f"Imposed load-point velocity, logarithmic from "
                        f"{SC.V_LP_MIN:.0e} to {SC.V_LP_MAX:.0e} m/s. "
                        f"PHYSICS: changing it continues the trajectory, it "
                        f"does not reset. Negative V_lp is not supported.")
                    ui.separator().props("vertical")
                    ui.label("Drag sensitivity").classes("text-xs")
                    ui.toggle(["low", "medium", "high"], value="medium",
                              on_change=lambda e:
                              self.c.set_manual_sensitivity(e.value)
                              ).props("dense")

                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Presets").classes("text-xs")
                    for key, lab in (("stable_vs", "Velocity strengthening"),
                                     ("stable_vw", "Stable velocity weakening"),
                                     ("near_critical", "Near critical"),
                                     ("unstable", "Stick-slip"),
                                     ("damping_matters",
                                      "Radiation damping matters")):
                        ui.button(lab,
                                  on_click=lambda _, k=key: self.on_preset(k)
                                  ).props("outline dense size=sm")
                    ui.separator().props("vertical")
                    self.lbl_sched = ui.label("").classes("text-xs text-grey-8")

    # ==================================================================
    # EVENT HANDLERS  (they only call the controller)
    # ==================================================================
    def on_play(self) -> None:
        self.c.play()

    def on_pause(self) -> None:
        self.c.pause()

    def on_step(self) -> None:
        self.c.single_step()

    def on_reset(self) -> None:
        self.c.reset()
        self._e_range = 1e-5
        self.err_banner.style("display:none")

    def on_hold(self) -> None:
        self.c.hold()
        self.sld_vlp.set_value(math.log10(SC.V_LP_MIN))

    def on_mode(self, e) -> None:
        self.manual_mode = bool(e.value)
        if self.manual_mode:
            self.c.pause()

    def on_vlp(self, e) -> None:
        if self.manual_mode:
            return
        self.c.set_vlp(10.0 ** float(e.value))

    def on_preset(self, key: str) -> None:
        summary = self.c.stage_preset(key)
        with ui.dialog() as dlg, ui.card():
            ui.label("Apply preset?").classes("text-bold")
            ui.label(summary).classes("text-xs v-num whitespace-pre-wrap")
            ui.label("Applying a preset performs a full Reset. Parameters "
                     "cannot be changed mid-trajectory: doing so would make "
                     "mu, tau, eta, kc and the spring preload discontinuous."
                     ).classes("text-xs text-orange-9")
            with ui.row():
                ui.button("Apply and Reset", on_click=lambda: (
                    self.c.apply_pending_preset(), self.on_reset_after(dlg)))
                ui.button("Cancel", on_click=lambda: (
                    self.c.cancel_pending_preset(), dlg.close())
                ).props("outline")
        dlg.open()

    def on_reset_after(self, dlg) -> None:
        self._e_range = 1e-5
        self.err_banner.style("display:none")
        self.sld_vlp.set_value(math.log10(max(self.c.V_lp_setpoint,
                                              SC.V_LP_MIN)))
        dlg.close()

    def on_mouse(self, e) -> None:
        """Pointer on the scene. The ONLY thing this can do is impose a
        load-point displacement. It cannot move the block."""
        if not self.manual_mode:
            return
        if e.type == "mousedown":
            self.c.begin_manual_pull(e.image_x)
        elif e.type == "mousemove":
            if self.c._manual_active:
                self.c.update_manual_pull(e.image_x,
                                          real_dt=1.0 / self.c.SCENE_HZ)
        elif e.type in ("mouseup", "mouseleave"):
            if self.c._manual_active:
                self.c.end_manual_pull()

    # ==================================================================
    # RENDER LOOPS
    # ==================================================================
    def tick_scene(self) -> None:
        try:
            s = self.c.snapshot()
        except Exception as exc:                                # noqa: BLE001
            self.lbl_state.set_text("interface read error")
            return
        try:
            # --- adaptive magnification, smoothed so it never jumps -------
            e_rel = abs(s.e - s.e0)
            self._e_range = max(self._e_range * 0.9995, e_rel, 1e-7)
            target = float(np.clip(110.0 / self._e_range, 1e2, 1e9))
            self.mag += 0.03 * (target - self.mag)      # slow EMA

            flash = s.event_flash_active
            if flash:
                self._shake_frames = 6
            shake = 0.0
            if self._shake_frames > 0:
                self._shake_frames -= 1
                shake = 3.0 * math.sin(self._shake_frames * 2.1)

            self.scene.set_content(
                build_scene_svg(s, self.mag, flash, shake, self.c.params))
            self.lbl_mag.set_text(
                f"Visual displacement exaggerated by x{self.mag:,.0f}   |   "
                f"actual: u_lp = {s.u_lp*1e6:,.3f} um, "
                f"delta = {s.delta*1e6:,.3f} um, "
                f"e = {s.e*1e6:,.3f} um  (e0 = {s.e0*1e6:,.3f} um)")
            self.lbl_manual.set_text(self.c.manual_warning or "")

            self.lbl_time.set_text(f"t = {s.t:,.4f} s"
                                   + ("" if s.live else "   [VIEWING PAST]"))
            self.lbl_state.set_text(s.visual_state_label)
            self.lbl_state.props(f'color={"deep-orange" if flash else "blue-grey-8"}')

            vals = dict(t=s.t, V_lp=s.V_lp, V=s.V, e=s.e,
                        tau_total=s.tau_total, mu=s.mu, theta=s.theta,
                        V_theta_over_Dc=s.V_theta_over_Dc)
            for key, _n, unit, _p, _m in EXPLORE_FIELDS:
                self.value_labels[key].set_text(_fmt(vals[key], unit))

            info = self.c.stability_info()
            self.lbl_kkc.set_text(
                f"{info['k_over_kc_qd']:.5f}" if info["kc_qd_defined"]
                else "not defined for this loading")
            self.lbl_op.set_text(s.physical_state_label)
            self.lbl_vlp.set_text(f"V_lp = {s.V_lp:.4e} m/s "
                                  f"= {s.V_lp*1e6:,.4g} um/s")
            self.lbl_sched.set_text(f"{info['schedule']}  |  mode: {s.mode}")

            if s.error:
                self.err_text.set_text(s.error)
                self.err_banner.style("display:block")

            self.update_physics(s, info)
        except Exception as exc:                                # noqa: BLE001
            # An interface failure is NOT a numerical failure. Say which.
            self.lbl_state.set_text("interface render error (engine is fine)")
            print(f"[interface] render error: {type(exc).__name__}: {exc}")

    def update_physics(self, s: SC.Snapshot, info: dict) -> None:
        p = self.c.params
        self.phys_params.set_text(
            p.summary(V_ss=info.get("V_ss")) + "\n\n"
            + f"evolution law : {self.c.law}\n"
            + f"loading       : {info['schedule']}\n\n"
            + info["kc_qd_note"])
        d = self.c.frame_dt_cap_detail()
        self.phys_num.set_text(
            f"tau_friction   = {s.tau_friction:,.4f} Pa\n"
            f"tau_damping    = {s.tau_damping:,.6f} Pa   (= eta*V)\n"
            f"tau_total      = {s.tau_total:,.4f} Pa\n"
            f"tau_spring     = {s.tau_spring:,.4f} Pa   (= k*e)\n"
            f"RESIDUAL       = {s.residual:.6e} Pa "
            f"({abs(s.residual)/p.sigma_n:.3e} of sigma_n)\n"
            f"                 analytically zero; this is integration error\n\n"
            f"solver success = {s.solver_success}\n"
            f"solver message = {s.solver_message or '(none)'}\n"
            f"nfev / njev    = {s.nfev} / {s.njev}   (latest step)\n"
            f"stored samples = {s.n_samples:,}\n"
            f"stale results discarded = {self.c.stale_results_discarded}\n\n"
            f"frame dt cap   = {d['cap']:.4e} s  "
            f"(binding: {d['binding']})\n"
            f"  slip bound   = {d['dt_slip']:.4e} s\n"
            f"  rate bound   = {d['dt_rate']:.4e} s   "
            f"(d(lnV)/dt = {d['dlnV_dt']:.4e} 1/s)")
        self.phys_state.set_text(
            f"PHYSICAL label : {s.physical_state_label}\n"
            f"  rule fired   : {s.op_rule}\n"
            f"VISUAL label   : {s.visual_state_label}\n"
            f"  flash active : {s.event_flash_active}"
            + (f"  ({s.event_flash_remaining_s:.2f} s of wall clock left of "
               f"{self.c.EVENT_FLASH_HOLD_S:.2f} s)"
               if s.event_flash_active else "")
            + f"\nevents catalogued : {s.n_events}"
            + (f"   last physical excursion (model time): "
               f"{self.c.last_visual_event[0]:.4f} to "
               f"{self.c.last_visual_event[1]:.4f} s, "
               f"V_max = {self.c.last_visual_event[2]:.4e} m/s"
               if self.c.last_visual_event else ""))

    def tick_plots(self) -> None:
        try:
            d = self.c.plot_data(n_max=1200)
            if d["t"].size < 2:
                return
            t = d["t"]
            marks = [{"xAxis": float(x)} for x in d["event_times"]]
            mline = {"symbol": "none", "silent": True,
                     "lineStyle": {"type": "dashed", "color": "#c2603a"},
                     "data": marks, "label": {"show": False}}

            def xy(y):
                return [[float(a), float(b)] for a, b in zip(t, y)]

            self.ch_v.options["series"] = [
                {"name": "V_lp [m/s]", "type": "line", "showSymbol": False,
                 "data": xy(np.maximum(d["V_lp"], 1e-12)),
                 "lineStyle": {"width": 1.6}, "markLine": mline},
                {"name": "V [m/s]", "type": "line", "showSymbol": False,
                 "data": xy(np.maximum(d["V"], 1e-12)),
                 "lineStyle": {"width": 1.6}},
            ]
            self.ch_tau.options["series"] = [
                {"name": "tau_total [MPa]", "type": "line",
                 "showSymbol": False, "data": xy(d["tau_total"] / 1e6),
                 "lineStyle": {"width": 1.6}, "markLine": mline},
                {"name": "tau_friction [MPa]", "type": "line",
                 "showSymbol": False, "data": xy(d["tau_friction"] / 1e6),
                 "lineStyle": {"width": 1.6}},
            ]
            self.ch_disp.options["series"] = [
                {"name": "u_lp [um]", "type": "line", "showSymbol": False,
                 "data": xy(d["u_lp"] * 1e6), "lineStyle": {"width": 1.6},
                 "markLine": mline},
                {"name": "delta [um]", "type": "line", "showSymbol": False,
                 "data": xy(d["delta"] * 1e6), "lineStyle": {"width": 1.6}},
            ]
            for ch in (self.ch_v, self.ch_tau, self.ch_disp):
                ch.update()
        except Exception as exc:                                # noqa: BLE001
            print(f"[interface] plot error (engine unaffected): "
                  f"{type(exc).__name__}: {exc}")


@ui.page("/")
def main_page() -> None:
    SpringSliderUI()


if __name__ in {"__main__", "__mp_main__"}:
    print("=" * 70)
    print(f" Interactive RSF spring-slider  (engine v{E.__version__}, "
          f"controller v{SC.__version__})")
    print(" Zero-dimensional, quasi-dynamic. Not a spatial fault model.")
    print(" Open http://127.0.0.1:8080")
    print("=" * 70)
    ui.run(title="RSF Spring-Slider", port=8080, reload=False,
           show=False, favicon="🪨")
