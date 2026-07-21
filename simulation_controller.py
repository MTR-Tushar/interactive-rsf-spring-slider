"""
=============================================================================
 simulation_controller.py   --   v0.2   (Stage 3)
=============================================================================
 The layer between the browser and rsf_engine.py.

 PURPOSE. Everything that is about OPERATING the simulation -- playing,
 pausing, choosing a loading mode, interpreting a mouse drag, labelling what
 the block is currently doing, remembering what to draw -- lives here. None of
 it is physics, and none of the physics lives here.

 THE RULE THIS FILE EXISTS TO ENFORCE:
     Browser code never touches the ODE state.
 The NiceGUI module calls methods on SimulationController and reads
 snapshot() / plot_data(). It has no access to lnV, theta or solve_ivp, and it
 cannot move the block. If you ever find yourself wanting to nudge a number to
 make the picture look right, that is the bug.

 EVERY DISPLAYED QUANTITY comes from rsf_engine.derived(), via History. The
 animation, the live values and the graphs all read the same arrays, so they
 cannot disagree with each other.

-----------------------------------------------------------------------------
 THREADING
-----------------------------------------------------------------------------
 Measured on this machine (see benchmark_engine.py), advance() costs about
 2-3 ms in the median across every regime and chunk size from 0.01 s to 5 s,
 but its TAIL during rapid slip reaches 51-182 ms. A 182 ms call on the UI
 thread is a five-frame freeze at 30 Hz, and it would happen exactly at the
 moment the user most wants to watch. Integration therefore runs in a worker
 thread (SimulationWorker) and the interface only ever reads snapshots.

 The controller is made thread-safe by a single re-entrant lock around all
 state mutation. The synchronous methods are fully usable without the worker,
 which is how the test suite drives it.

-----------------------------------------------------------------------------
 MODEL TIME vs REAL TIME  (and why one speed cannot show everything)
-----------------------------------------------------------------------------
 A stick-slip cycle at the default parameters lasts ~774 s of model time; the
 event inside it lasts ~0.035 s. That is a ratio of ~2e4. No single playback
 speed can show both: at a speed that renders the cycle in half a minute, the
 whole event happens inside a single frame and is invisible.

 So the controller separates two things that must never be confused:

   speed_multiplier   MODEL SECONDS PER REAL SECOND. A playback/display
                      setting. It changes how fast you watch. It has NO
                      effect on V_lp, on the equations, or on the solution.

   V_lp               The imposed physical load-point velocity [m/s]. This
                      is physics.

 and it offers an optional, clearly-labelled EVENT SLOW-MOTION: when V rises
 above a threshold, the model time advanced per frame is capped, so the event
 unfolds over seconds of real time instead of one frame. This is still purely
 a display-rate choice -- it changes how much model time each frame covers,
 not what the solution is. Chunking was shown in Stage 2 to leave the
 interseismic solution unchanged to 1 part in 1e13.
=============================================================================
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import rsf_engine as E
from rsf_engine import (ConstantLoading, EventDetector, History,
                        LoadingSchedule, RSFParams, SimulationState,
                        SpringSliderEngine)

__version__ = "0.2"


# =============================================================================
# OPERATIONAL STATE LABELS
# =============================================================================
#
# THESE ARE INTERFACE CONVENTIONS, NOT PHYSICS.
#
# The project text is explicit that the four-stage decomposition of a seismic
# cycle is a simplification: preparatory slip may be weak or absent as an
# observationally distinct phase, nucleation is model-defined, and transitions
# may be gradual rather than sharp. The labels below are a reading aid for
# someone watching an animation. They are NOT claims about universal phases of
# faulting, and a different threshold set would relabel the same solution.
#
# All thresholds live in this ONE place so that they can be found, cited and
# changed. The Physics tab displays the exact rule that fired.
# =============================================================================

@dataclass(frozen=True)
class StateThresholds:
    """Every number used to classify what the block is doing. All SI."""

    # An excursion above this is called an event. Same operational convention
    # as the engine's event detector; kept equal to it by default.
    V_event: float = 1e-3                 # [m/s]

    # V/V_lp band counted as "keeping up with the load point".
    creep_ratio_lo: float = 0.5           # [-]
    creep_ratio_hi: float = 2.0           # [-]

    # Below this fraction of V_lp the interface is called nearly locked.
    locked_ratio: float = 0.1             # [-]

    # Absolute fallback for Hold (V_lp = 0), where a ratio is meaningless.
    V_locked_abs: float = 1e-9            # [m/s]

    # d(ln V)/dt above this counts as accelerating, but ONLY when the block
    # is already outrunning the load point (V > creep_ratio_hi * V_lp). The
    # gate matters: an unstable spring-slider grows exponentially at
    # Re(lambda) ~ 1.5e-2 1/s throughout its whole interseismic phase, and a
    # bare rate test would therefore label a block that is merely settling
    # back onto steady creep as "Accelerating". Requiring the block to be
    # pulling away from the load point as well makes the label mean what a
    # learner would expect it to mean.
    dlnV_dt_accel: float = 1e-3           # [1/s]

    # Model time after an event ends during which the label stays
    # "Post-event relaxation".
    post_event_window: float = 20.0       # [s]


THRESHOLDS = StateThresholds()


class OpState(str, Enum):
    LOCKED = "Nearly locked"
    CREEP = "Stable creep"
    ACCELERATING = "Accelerating"
    EVENT = "Rapid-slip event"
    RELAXING = "Post-event relaxation"
    TRANSIENT = "Transient"
    UNKNOWN = "No data yet"


def classify_state(V: float, V_lp: float, dlnV_dt: float,
                   t: float, t_last_event_end: Optional[float],
                   thr: StateThresholds = THRESHOLDS) -> Tuple[OpState, str]:
    """Return (label, the exact rule that produced it).

    Rules are tested in order; the first match wins. The returned rule string
    is shown verbatim in the Physics tab so the label is never mysterious.
    """
    if not np.isfinite(V):
        return OpState.UNKNOWN, "V is not finite"

    if V > thr.V_event:
        return (OpState.EVENT,
                f"V = {V:.3e} m/s > V_event = {thr.V_event:.1e} m/s")

    if (t_last_event_end is not None
            and 0.0 <= t - t_last_event_end < thr.post_event_window):
        return (OpState.RELAXING,
                f"{t - t_last_event_end:.3g} s since the last event ended "
                f"< post_event_window = {thr.post_event_window:.3g} s")

    outrunning = (V > thr.creep_ratio_hi * V_lp if V_lp > 0.0
                  else V > thr.V_locked_abs)
    if dlnV_dt > thr.dlnV_dt_accel and outrunning:
        gate = (f"V > {thr.creep_ratio_hi} * V_lp" if V_lp > 0.0
                else f"V > V_locked_abs (V_lp = 0)")
        return (OpState.ACCELERATING,
                f"d(ln V)/dt = {dlnV_dt:.3e} 1/s > dlnV_dt_accel = "
                f"{thr.dlnV_dt_accel:.1e} 1/s AND {gate}")

    if V_lp > 0.0:
        ratio = V / V_lp
        if thr.creep_ratio_lo <= ratio <= thr.creep_ratio_hi:
            return (OpState.CREEP,
                    f"V/V_lp = {ratio:.4g} is inside "
                    f"[{thr.creep_ratio_lo}, {thr.creep_ratio_hi}]")
        if ratio < thr.locked_ratio:
            return (OpState.LOCKED,
                    f"V/V_lp = {ratio:.3e} < locked_ratio = "
                    f"{thr.locked_ratio}")
        return (OpState.TRANSIENT,
                f"V/V_lp = {ratio:.4g} is outside the creep band but not "
                f"below locked_ratio = {thr.locked_ratio}")

    # Hold: V_lp = 0, so a ratio is undefined and we fall back to absolute V.
    if V < thr.V_locked_abs:
        return (OpState.LOCKED,
                f"V_lp = 0 (Hold) and V = {V:.3e} m/s < V_locked_abs = "
                f"{thr.V_locked_abs:.1e} m/s")
    return (OpState.TRANSIENT,
            f"V_lp = 0 (Hold) and V = {V:.3e} m/s >= V_locked_abs = "
            f"{thr.V_locked_abs:.1e} m/s")


# =============================================================================
# LOADING MODES AND MANUAL PULL
# =============================================================================

class LoadingMode(str, Enum):
    VELOCITY = "velocity control"
    HOLD = "hold (V_lp = 0)"
    MANUAL = "manual pull"


# V_lp control range. The lower end is set by patience rather than by
# numerics: at 1e-9 m/s the default cycle would take ~7.7e5 s of model time.
# The upper end is where radiation damping starts to matter visibly
# (eta*V_lp = 5e3 Pa at 1e-3 m/s, i.e. 10% of sigma_n(b-a)) and where kc_QD
# departs measurably from kc_QS. Both ends were checked against the engine.
V_LP_MIN = 1e-9      # [m/s]
V_LP_MAX = 1e-3      # [m/s]

# Hard ceiling on the V_lp a mouse drag may impose. Dragging faster than this
# does NOT truncate the displacement -- see ManualPullConfig.
V_LP_MANUAL_MAX = 1e-3   # [m/s]


@dataclass
class ManualPullConfig:
    """Mapping from screen pixels to imposed load-point displacement.

    metres_per_pixel is displayed in the interface at all times, because a
    drag whose physical meaning is hidden is a decorative drag.

    WHAT HAPPENS IF YOU DRAG TOO FAST. The imposed displacement is honoured
    EXACTLY; it is the model-time interval that is stretched. If a segment
    would need V_lp above v_lp_max, the segment is instead integrated over a
    longer model time so that V_lp = v_lp_max, and the controller raises a
    'saturated' warning. The alternative -- truncating the displacement --
    would silently disagree with where you put the mouse, and would break the
    guarantee that final u_lp equals the total imposed displacement.
    """
    metres_per_pixel: float = 2e-6        # [m/px]  "medium"
    v_lp_max: float = V_LP_MANUAL_MAX     # [m/s]
    release_behaviour: str = "hold"       # documented default on pointer-up

    SENSITIVITIES = {"low": 2e-7, "medium": 2e-6, "high": 2e-5}   # [m/px]

    def describe(self) -> str:
        return (f"{self.metres_per_pixel:.3e} m of load-point displacement "
                f"per screen pixel "
                f"({self.metres_per_pixel*1e6:.4g} um/px)")


# =============================================================================
# SNAPSHOT
# =============================================================================

@dataclass
class Snapshot:
    """Everything the interface needs for one frame. Plain numbers only, so
    the UI cannot accidentally hold a reference to solver internals."""
    ok: bool
    t: float
    V_lp: float
    V: float
    theta: float
    delta: float
    u_lp: float
    e: float
    e0: float
    mu: float
    tau_friction: float
    tau_damping: float
    tau_total: float
    tau_spring: float
    residual: float
    V_theta_over_Dc: float
    # PHYSICAL classification, computed from the stored history sample.
    # This is the scientific label and the visual latch never modifies it.
    op_state: str
    op_rule: str
    physical_state_label: str      # explicit alias of op_state
    # VISUAL label actually painted on the scene. Equals op_state except
    # while the display-only event latch is held. See EVENT_FLASH_HOLD_S.
    visual_state_label: str
    event_flash_active: bool
    event_flash_remaining_s: float
    mode: str
    playing: bool
    live: bool                 # False when viewing a past time
    speed_multiplier: float
    n_events: int
    last_event_t: Optional[float]
    error: Optional[str]
    solver_message: str
    solver_success: bool
    nfev: int
    njev: int
    n_samples: int
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# THE CONTROLLER
# =============================================================================

class SimulationController:
    """UI-independent operator for the spring-slider engine."""

    #: model time advanced by one press of Single Step [s]
    SINGLE_STEP_DT = 0.5

    #: playback presets, in MODEL SECONDS PER REAL SECOND
    SPEEDS = {"Slow": 1.0, "Normal": 25.0, "Fast": 200.0}

    # ---------------------------------------------------------------
    # ADAPTIVE ADVANCEMENT
    # ---------------------------------------------------------------
    # A FIXED model time per frame cannot work, and this was measured rather
    # than assumed. At the default parameters the loading cycle is ~774 s and
    # the rapid-slip event is ~0.0338 s: a ratio of 2.3e4. The nucleation ramp
    # from V = V_lp up to the 1 mm/s threshold takes 11.7 s of model time, so
    # a single "Fast" frame of 8 s covers 0.7 of the ENTIRE ramp and lands on
    # the far side of the event. Measured directly: 0 of 45 UI frames sampled
    # V above threshold, while History held 451 samples inside the event with
    # a largest gap of 1.7e-4 s. The solver was never at fault; the frame
    # sampling was.
    #
    # So the model time per frame is capped by two signals, both measured:
    #
    # (1) SLIP PER FRAME. dt <= SLIP_FRACTION_PER_FRAME * Dc / max(V, V_lp).
    #     Dc is the distance over which state evolves, so limiting slip per
    #     frame to a fraction of Dc guarantees each frame resolves state
    #     evolution. Interseismically V < V_lp so the bound is set by V_lp and
    #     is loose (5 s at the defaults); during rapid slip it tightens
    #     automatically in proportion to V.
    #
    # (2) LOG-VELOCITY CHANGE PER FRAME. dt <= MAX_DLNV_PER_FRAME / |dlnV/dt|.
    #     Measured d(ln V)/dt at the defaults: 0.013 1/s interseismically,
    #     0.125 1/s at V ~ V_lp, 15.5 1/s at V ~ 1e-4, 183 1/s in the event.
    #     A four-decade signal, so this bound is inert during quiet loading
    #     and clamps hard on the approach to instability -- which is exactly
    #     the phase the slip bound alone is too loose to catch.
    #
    # Neither bound touches the equations. They change how much model time one
    # FRAME covers, i.e. the chunk size, and Stage 2 measured chunking to be
    # neutral to 1 part in 1e13 interseismically.
    SLIP_FRACTION_PER_FRAME = 0.25   # [-] slip per frame, in units of Dc
    MAX_DLNV_PER_FRAME = 0.35        # [-] V changes by at most e^0.35 = 1.42x
    MIN_FRAME_DT = 1e-7              # [s] floor, so the loop cannot stall

    # ---------------------------------------------------------------
    # VISUAL EVENT LATCH  --  A DISPLAY RULE, NOT PHYSICS
    # ---------------------------------------------------------------
    # An event lasting 0.0338 s of model time need not coincide with any UI
    # refresh. The latch keeps the flash and the "Rapid-slip event" visual
    # label alive for a minimum of EVENT_FLASH_HOLD_S of WALL-CLOCK time after
    # the physical event, so a real event is never invisible.
    #
    # It must not, and does not: alter the trajectory; lengthen the physical
    # event duration; change the event catalogue; claim V is still above
    # threshold; or alter the operational classification stored in the
    # scientific history. The physical label and the visual label are exposed
    # as SEPARATE fields and the Physics view shows both.
    EVENT_FLASH_HOLD_S = 0.6         # [s of REAL time]

    #: Frame rates chosen from benchmark_engine.py (see module docstring).
    SCENE_HZ = 25
    PLOT_HZ = 4

    def __init__(self, preset_name: str = "unstable",
                 V_lp: float = 1e-6, law: str = "ageing"):
        # _lock guards controller FIELDS. It is deliberately NOT held while
        # solve_ivp runs: integration has a measured tail of 182 ms, and
        # holding the lock across it would block the UI thread's snapshot()
        # reads for exactly as long -- reintroducing the freeze the worker
        # thread exists to prevent.
        self._lock = threading.RLock()
        # _integration_lock guarantees that at most ONE integration job is in
        # flight, whatever calls advance. Duplicate Play presses, a stray
        # Single Step during a worker frame, and the worker itself all
        # serialise here.
        self._integration_lock = threading.Lock()
        # Generation ID. Incremented by every Reset. A result computed against
        # generation N is DISCARDED if it comes back after the generation has
        # moved on, so a Reset can never be overwritten by an integration that
        # was already running when it happened.
        self._generation = 0
        self.stale_results_discarded = 0
        self._preset_name = preset_name
        self._pending_preset: Optional[str] = None
        self.law = law
        self.speed_name = "Normal"
        self.adaptive_advance = True
        self.manual = ManualPullConfig()
        self._reset_internal(preset_name=preset_name, V_lp=V_lp, law=law)

    # ------------------------------------------------------------------
    # RESET AND PRESETS
    # ------------------------------------------------------------------
    def _bump_generation(self) -> None:
        self._generation += 1

    def _reset_internal(self, preset_name: Optional[str] = None,
                        params: Optional[RSFParams] = None,
                        V_lp: Optional[float] = None,
                        law: Optional[str] = None) -> None:
        if preset_name is not None:
            p, sched = E.preset(preset_name)
            self._preset_name = preset_name
            if V_lp is None:
                V_lp = sched.steady_V() or 1e-6
        else:
            p = params if params is not None else self.params
            V_lp = V_lp if V_lp is not None else 1e-6
        if law is not None:
            self.law = law

        self.params = p
        self.mode = LoadingMode.VELOCITY
        self.V_lp_setpoint = float(V_lp)
        self.engine = SpringSliderEngine(p, ConstantLoading(self.V_lp_setpoint),
                                         law=self.law)
        # Start slightly off the fixed point: a perfectly steady start is a
        # fixed point and nothing would ever happen.
        self.state = self.engine.initial_state(V_fraction=0.9)
        self.history = History()
        self.history.append_initial(self.engine, self.state)
        self.detector = EventDetector(THRESHOLDS.V_event)
        self.playing = False
        self.error: Optional[str] = None
        self.last_report: Optional[E.StepReport] = None
        self._t_last_event_end: Optional[float] = None
        self._selected_time: Optional[float] = None
        self._manual_active = False
        self._manual_last_px: Optional[float] = None
        self._manual_total_u: float = 0.0
        self.manual_warning: Optional[str] = None
        self._pending_preset = None
        self._event_flash_until = 0.0        # wall clock, time.monotonic()
        self._last_visual_event: Optional[Tuple[float, float, float]] = None
        self._pending_schedule: Optional[LoadingSchedule] = None
        self._bump_generation()

    def reset(self, preset_name: Optional[str] = None,
              params: Optional[RSFParams] = None,
              V_lp: Optional[float] = None,
              law: Optional[str] = None) -> None:
        """Full Reset. This is the ONLY way to change a, b, Dc, sigma_n, V0,
        G, cs, k or the evolution law: those change the governing model and
        would create discontinuities in mu, tau, eta, kc or the spring preload
        if applied mid-trajectory."""
        with self._lock:
            if preset_name is None and params is None:
                preset_name = self._preset_name
            self._reset_internal(preset_name=preset_name, params=params,
                                 V_lp=V_lp, law=law)

    def stage_preset(self, name: str) -> str:
        """Select a preset WITHOUT applying it. Pauses and returns a summary.
        The trajectory is untouched until apply_pending_preset() is called,
        which performs a Reset. Presets are never swapped in silently."""
        with self._lock:
            if name not in E.PRESET_NAMES:
                raise ValueError(f"unknown preset '{name}'")
            self.playing = False
            self._pending_preset = name
            p, sched = E.preset(name)
            V_ss = sched.steady_V()
            head = (f"PRESET '{name}' -- NOT YET APPLIED.\n"
                    f"Applying it will RESET the current trajectory "
                    f"(t = {self.state.t:.3f} s, "
                    f"{len(self.detector.events)} events so far).\n\n")
            return head + p.summary(V_ss=V_ss)

    def has_pending_preset(self) -> bool:
        with self._lock:
            return self._pending_preset is not None

    def apply_pending_preset(self) -> None:
        with self._lock:
            if self._pending_preset is None:
                return
            self.reset(preset_name=self._pending_preset)

    def cancel_pending_preset(self) -> None:
        with self._lock:
            self._pending_preset = None

    # ------------------------------------------------------------------
    # PLAYBACK
    # ------------------------------------------------------------------
    def play(self) -> None:
        with self._lock:
            if self.error is None:
                self.playing = True
                self._selected_time = None      # resume live

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def toggle_play(self) -> None:
        with self._lock:
            (self.pause if self.playing else self.play)()

    def set_speed(self, name: str) -> None:
        """Change the PLAYBACK rate. This is a display setting: it does not
        touch V_lp, the parameters or the equations."""
        with self._lock:
            if name not in self.SPEEDS:
                raise ValueError(f"unknown speed '{name}'")
            self.speed_name = name

    @property
    def speed_multiplier(self) -> float:
        """Model seconds per real second."""
        return self.SPEEDS[self.speed_name]

    def single_step(self, dt_model: Optional[float] = None) -> bool:
        """Advance exactly dt_model seconds of MODEL time (default
        SINGLE_STEP_DT) and pause. Returns True on success."""
        with self._lock:
            self.playing = False
            dt = (self.SINGLE_STEP_DT if dt_model is None
                  else float(dt_model))
        return self._advance_model_time(dt)

    def advance_frame(self, real_dt: Optional[float] = None) -> bool:
        """Advance one animation frame's worth of model time.

        real_dt is the wall-clock interval this frame represents; the model
        time advanced is real_dt * speed_multiplier, optionally capped by
        event slow-motion. Returns True on success (or if not playing).
        """
        with self._lock:
            if not self.playing or self.error is not None:
                return True
            if real_dt is None:
                real_dt = 1.0 / self.SCENE_HZ
            dt = real_dt * self.speed_multiplier
            if self.adaptive_advance:
                dt = min(dt, self.frame_dt_cap())
        # Lock released before integrating -- see __init__.
        return self._advance_model_time(dt)

    def frame_dt_cap(self) -> float:
        """Largest model time one frame may cover, from the two measured
        signals documented at the top of this class. Public so the Physics
        view can show the number and say which bound is active."""
        p = self.params
        V = self.state.V
        V_lp = self.current_V_lp()
        dt_slip = self.SLIP_FRACTION_PER_FRAME * p.Dc / max(V, V_lp, 1e-30)
        rate = abs(self._dlnV_dt())
        dt_rate = (self.MAX_DLNV_PER_FRAME / rate if rate > 0 else np.inf)
        return max(min(dt_slip, dt_rate), self.MIN_FRAME_DT)

    def frame_dt_cap_detail(self) -> Dict[str, Any]:
        p = self.params
        V = self.state.V
        V_lp = self.current_V_lp()
        dt_slip = self.SLIP_FRACTION_PER_FRAME * p.Dc / max(V, V_lp, 1e-30)
        rate = abs(self._dlnV_dt())
        dt_rate = (self.MAX_DLNV_PER_FRAME / rate if rate > 0 else np.inf)
        return dict(dt_slip=dt_slip, dt_rate=dt_rate, dlnV_dt=self._dlnV_dt(),
                    cap=self.frame_dt_cap(),
                    binding=("slip-per-frame" if dt_slip <= dt_rate
                             else "d(lnV)/dt-per-frame"),
                    enabled=self.adaptive_advance)

    # ------------------------------------------------------------------
    # THE ONE PLACE THE ENGINE IS STEPPED
    # ------------------------------------------------------------------
    def _advance_model_time(self, dt: float) -> bool:
        """All model-time advancement funnels through here.

        Three-phase, so that the expensive part runs without the field lock:
          1. under _lock: read the generation, apply any pending schedule
             change, take references to the engine and state;
          2. lock released: run solve_ivp (the 2-182 ms part);
          3. under _lock: if the generation still matches, apply the result;
             otherwise a Reset happened while we were integrating and the
             result is STALE and discarded.
        _integration_lock ensures only one job is ever in flight.
        """
        if dt <= 0:
            return True

        with self._integration_lock:
            with self._lock:
                gen = self._generation
                if self._pending_schedule is not None:
                    # Loading changes take effect exactly at a chunk boundary,
                    # which is here: immediately before a fresh integration.
                    self.engine.set_schedule(self._pending_schedule)
                    self._pending_schedule = None
                engine = self.engine
                state = self.state

            try:
                new_state, samples, report = engine.advance(state, dt)
            except Exception as exc:                            # noqa: BLE001
                with self._lock:
                    if self._generation != gen:
                        self.stale_results_discarded += 1
                        return True
                    self.playing = False
                    self.error = (
                        f"Integration raised {type(exc).__name__}: {exc}. "
                        f"Mode = {self.mode.value}, V_lp = "
                        f"{self.current_V_lp():.4e} m/s. The trajectory up to "
                        f"t = {self.state.t:.6g} s is preserved. Reset to "
                        f"continue.")
                return False

            with self._lock:
                if self._generation != gen:
                    # A Reset landed while this was integrating. The result
                    # belongs to a trajectory that no longer exists.
                    self.stale_results_discarded += 1
                    return True

                self.last_report = report
                # Record whatever WAS computed, even on failure: a partial
                # trajectory is real data and must not be thrown away.
                if samples["t"].size:
                    self.history.append(samples)
                    self.detector.feed(samples)
                    done = [e for e in self.detector.events
                            if not e.truncated]
                    if done:
                        self._t_last_event_end = done[-1].t_end
                    # VISUAL LATCH. Armed if ANY sample in this chunk exceeded
                    # the event threshold, which catches an excursion that
                    # began and ended entirely inside one frame. Purely a
                    # display timer: the samples, the catalogue and the stored
                    # classification are untouched by it.
                    above = samples["V"] > THRESHOLDS.V_event
                    if bool(above.any()):
                        self._event_flash_until = (time.monotonic()
                                                   + self.EVENT_FLASH_HOLD_S)
                        tt, vv = samples["t"][above], samples["V"][above]
                        self._last_visual_event = (float(tt[0]),
                                                   float(tt[-1]),
                                                   float(vv.max()))
                self.state = new_state

                if not report.success:
                    self.playing = False
                    self.error = (
                        f"{report.message}\n"
                        f"Mode = {self.mode.value}, V_lp = "
                        f"{self.current_V_lp():.4e} m/s, law = {self.law}, "
                        f"k/kc_QS = "
                        f"{self.params.stiffness_ratio_qs():.4f}. The "
                        f"trajectory up to t = {self.state.t:.6g} s is "
                        f"preserved and is real. Nothing beyond it has been "
                        f"invented. Press Reset to start a new run.")
                    return False
        return True

    # ------------------------------------------------------------------
    # LOADING CONTROL
    # ------------------------------------------------------------------
    def current_V_lp(self) -> float:
        """The V_lp that IS or WILL BE imposed. If a change is pending it is
        reported here, because it is what the next step will use and what the
        user just asked for."""
        sched = self._pending_schedule or self.engine.schedule
        return float(sched.v_lp(self.state.t))

    def set_vlp(self, V_lp: float) -> None:
        """Change the imposed load-point velocity, continuing from the
        current state. Takes effect at the next chunk boundary, which is the
        next advance() call. Does NOT reset, and does NOT alter V, theta,
        delta, u_lp, e0 or any stress."""
        with self._lock:
            V_lp = float(V_lp)
            if V_lp < 0:
                raise ValueError("Negative V_lp is not supported: reverse "
                                 "loading needs a signed-velocity friction "
                                 "formulation, which is outside this model.")
            V_lp = float(np.clip(V_lp, 0.0, V_LP_MAX))
            self.V_lp_setpoint = V_lp
            self.mode = LoadingMode.HOLD if V_lp == 0.0 else LoadingMode.VELOCITY
            self._pending_schedule = ConstantLoading(V_lp)

    def hold(self) -> None:
        """Impose V_lp = 0. The load point stops; the block does not."""
        with self._lock:
            self.mode = LoadingMode.HOLD
            self._pending_schedule = ConstantLoading(0.0)

    # ------------------------------------------------------------------
    # MANUAL PULL
    # ------------------------------------------------------------------
    def set_manual_sensitivity(self, name: str) -> None:
        with self._lock:
            if name not in ManualPullConfig.SENSITIVITIES:
                raise ValueError(f"unknown sensitivity '{name}'")
            self.manual.metres_per_pixel = \
                ManualPullConfig.SENSITIVITIES[name]

    def begin_manual_pull(self, x_px: float) -> None:
        """Pointer down on the load-point handle."""
        with self._lock:
            self.playing = False
            self.mode = LoadingMode.MANUAL
            self._manual_active = True
            self._manual_last_px = float(x_px)
            self._manual_total_u = 0.0
            self.manual_warning = None

    def update_manual_pull(self, x_px: float,
                           real_dt: Optional[float] = None) -> bool:
        """Pointer move. Converts the movement into an imposed load-point
        displacement, derives the segment's V_lp, and integrates the ENGINE
        over that segment. The block position that results is whatever the
        RSF equations produce -- it is never set from the pointer.

        Returns True on success.
        """
        with self._lock:
            if not self._manual_active or self._manual_last_px is None:
                return True
            x_px = float(x_px)
            dx_px = x_px - self._manual_last_px
            du = dx_px * self.manual.metres_per_pixel      # [m]

            if du <= 0.0:
                # Backward drag. Ignored, not converted into negative loading.
                self._manual_last_px = x_px
                if du < 0.0:
                    self.manual_warning = (
                        "Backward drag ignored. Reverse loading is not "
                        "supported: the logarithmic friction law requires "
                        "V > 0 and a signed-velocity formulation would be a "
                        "different model. The load point does not move back.")
                return True

            if real_dt is None:
                real_dt = 1.0 / self.SCENE_HZ
            dt_model = real_dt * self.speed_multiplier

            # Honour the displacement exactly; stretch model time if the drag
            # is faster than the ceiling allows. See ManualPullConfig.
            v_seg = du / dt_model if dt_model > 0 else np.inf
            saturated = v_seg > self.manual.v_lp_max
            if saturated:
                v_seg = self.manual.v_lp_max
                dt_model = du / v_seg
                self.manual_warning = (
                    f"Drag speed saturated at V_lp = {v_seg:.3e} m/s. The "
                    f"displacement you asked for is applied in full, but it "
                    f"is spread over {dt_model:.4g} s of model time instead "
                    f"of {real_dt * self.speed_multiplier:.4g} s. Model time "
                    f"is running slower than the playback setting for this "
                    f"segment.")
            else:
                self.manual_warning = None

            self._pending_schedule = ConstantLoading(v_seg)
            self._manual_last_px = x_px

        # Lock released before integrating, as everywhere else.
        ok = self._advance_model_time(dt_model)
        if ok:
            with self._lock:
                self._manual_total_u += du
        return ok

    def end_manual_pull(self) -> None:
        """Pointer up. Defaults to Hold, as documented."""
        with self._lock:
            self._manual_active = False
            self._manual_last_px = None
            if self.manual.release_behaviour == "hold":
                self.mode = LoadingMode.HOLD
                self._pending_schedule = ConstantLoading(0.0)

    @property
    def manual_total_displacement(self) -> float:
        """Total load-point displacement imposed by drags since the drag
        began [m]. Compared against u_lp in the tests."""
        return self._manual_total_u

    # ------------------------------------------------------------------
    # TIMELINE
    # ------------------------------------------------------------------
    def select_history_time(self, t: float) -> None:
        """Pause and view a past time. The picture is reconstructed from
        STORED history, never by re-running the model."""
        with self._lock:
            self.playing = False
            self._selected_time = float(t)

    def return_to_live(self) -> None:
        with self._lock:
            self._selected_time = None

    @property
    def is_live(self) -> bool:
        return self._selected_time is None

    # ------------------------------------------------------------------
    # VISUAL EVENT LATCH (display only)
    # ------------------------------------------------------------------
    @property
    def event_flash_active(self) -> bool:
        """True while the flash should be shown. Wall-clock, not model time.
        Never true while scrubbing: there the picture is a past moment and
        the honest label is whatever was happening then."""
        with self._lock:
            if self._selected_time is not None:
                return False
            return time.monotonic() < self._event_flash_until

    @property
    def event_flash_remaining_s(self) -> float:
        with self._lock:
            return max(0.0, self._event_flash_until - time.monotonic())

    @property
    def last_visual_event(self) -> Optional[Tuple[float, float, float]]:
        """(t_first, t_last, V_max) of the last above-threshold excursion seen
        in a frame. Model times, read straight from stored samples."""
        with self._lock:
            return self._last_visual_event

    # ------------------------------------------------------------------
    # READ-ONLY OUTPUT FOR THE INTERFACE
    # ------------------------------------------------------------------
    def _dlnV_dt(self) -> float:
        """Finite-difference d(ln V)/dt from the last two stored samples."""
        a = self.history.arrays()
        if a["t"].size < 2:
            return 0.0
        dt = a["t"][-1] - a["t"][-2]
        if dt <= 0:
            return 0.0
        return float((np.log(a["V"][-1]) - np.log(a["V"][-2])) / dt)

    def snapshot(self) -> Snapshot:
        """One frame of everything the interface displays."""
        with self._lock:
            a = self.history.arrays()
            if a["t"].size == 0:
                return Snapshot(
                    ok=False, t=0, V_lp=0, V=0, theta=0, delta=0, u_lp=0, e=0,
                    e0=0, mu=0, tau_friction=0, tau_damping=0, tau_total=0,
                    tau_spring=0, residual=0, V_theta_over_Dc=0,
                    op_state=OpState.UNKNOWN.value, op_rule="no samples",
                    physical_state_label=OpState.UNKNOWN.value,
                    visual_state_label=OpState.UNKNOWN.value,
                    event_flash_active=False, event_flash_remaining_s=0.0,
                    mode=self.mode.value, playing=self.playing, live=True,
                    speed_multiplier=self.speed_multiplier, n_events=0,
                    last_event_t=None, error=self.error, solver_message="",
                    solver_success=True, nfev=0, njev=0, n_samples=0)

            if self._selected_time is None:
                i = len(self.history) - 1
                dlnV = self._dlnV_dt()
            else:
                i = self.history.index_at_time(self._selected_time)
                if i >= 1:
                    dt = a["t"][i] - a["t"][i - 1]
                    dlnV = (float((np.log(a["V"][i]) - np.log(a["V"][i - 1]))
                                  / dt) if dt > 0 else 0.0)
                else:
                    dlnV = 0.0

            snap = self.history.snapshot(i)

            # V_lp DISPLAY CONVENTION. When live, show the value the engine
            # is imposing RIGHT NOW, not the value recorded at the last
            # integrated sample. Otherwise pressing Hold leaves the readout
            # showing the old velocity until the next step happens -- and
            # while paused that step never comes, so the display would
            # contradict the button the user just pressed. When scrubbing the
            # timeline we show the historical value instead, because there
            # the question being asked is "what was V_lp then?".
            V_lp_shown = (self.current_V_lp() if self._selected_time is None
                          else snap["V_lp"])

            op, rule = classify_state(
                snap["V"], V_lp_shown, dlnV, snap["t"],
                self._t_last_event_end)
            rep = self.last_report
            flash = (self._selected_time is None
                     and time.monotonic() < self._event_flash_until)
            done = [e for e in self.detector.events if not e.truncated]
            return Snapshot(
                ok=self.error is None,
                t=snap["t"], V_lp=V_lp_shown, V=snap["V"],
                theta=snap["theta"], delta=snap["delta"], u_lp=snap["u_lp"],
                e=snap["e"], e0=self.state.e0, mu=snap["mu"],
                tau_friction=snap["tau_friction"],
                tau_damping=snap["tau_damping"],
                tau_total=snap["tau_total"], tau_spring=snap["tau_spring"],
                residual=snap["residual"],
                V_theta_over_Dc=snap["V_theta_over_Dc"],
                op_state=op.value, op_rule=rule,
                physical_state_label=op.value,
                visual_state_label=(OpState.EVENT.value if flash
                                    else op.value),
                event_flash_active=flash,
                event_flash_remaining_s=self.event_flash_remaining_s,
                mode=self.mode.value,
                playing=self.playing, live=self._selected_time is None,
                speed_multiplier=self.speed_multiplier,
                n_events=len(self.detector.events),
                last_event_t=done[-1].t_start if done else None,
                error=self.error,
                solver_message=rep.message if rep else "",
                solver_success=rep.success if rep else True,
                nfev=rep.nfev if rep else 0, njev=rep.njev if rep else 0,
                n_samples=len(self.history),
                warnings=list(rep.warnings) if rep else [])

    def plot_data(self, n_max: int = 1500) -> Dict[str, Any]:
        """Decimated arrays for plotting, plus event onset times.

        Decimation is FOR DISPLAY ONLY. The full history is retained and is
        what CSV export and any statistics use; decimating first would remove
        the peaks of short events.
        """
        with self._lock:
            d = self.history.decimate(n_max)
            return dict(
                t=d["t"], V=d["V"], V_lp=d["V_lp"],
                tau_total=d["tau_total"], tau_friction=d["tau_friction"],
                tau_damping=d["tau_damping"], u_lp=d["u_lp"],
                delta=d["delta"], e=d["e"], mu=d["mu"], theta=d["theta"],
                event_times=[e.t_start for e in self.detector.events],
                n_full=len(self.history), n_shown=int(d["t"].size))

    def stability_info(self) -> Dict[str, Any]:
        """kc_QS, kc_QD and the ratios -- with an explicit statement when a
        stationary kc_QD is not defined."""
        with self._lock:
            p = self.params
            sched = self.engine.schedule
            info: Dict[str, Any] = dict(
                k=p.k, kc_qs=p.kc_qs, k_over_kc_qs=p.stiffness_ratio_qs(),
                law=self.law, mode=self.mode.value,
                schedule=sched.describe())
            V_ss = sched.steady_V() if sched.is_stationary() else None
            if V_ss is not None and self.mode is LoadingMode.VELOCITY:
                info.update(
                    kc_qd=p.kc_qd(V_ss), k_over_kc_qd=p.stiffness_ratio_qd(V_ss),
                    V_ss=V_ss, kc_qd_defined=True,
                    kc_qd_note=(
                        "Constant loading, so a steady-sliding state exists "
                        "and kc_QD applies. The implemented (quasi-dynamic) "
                        "system follows kc_QD, not kc_QS."))
            else:
                info.update(
                    kc_qd=None, k_over_kc_qd=None, V_ss=None,
                    kc_qd_defined=False,
                    kc_qd_note=(
                        "NO STATIONARY kc_QD IS DEFINED for this loading. "
                        "kc_QD = [sigma_n(b-a) - eta*V_ss]/Dc requires a "
                        "steady-sliding velocity V_ss, which exists only for "
                        "constant positive V_lp. Under Hold, manual pull, or "
                        "any time-dependent schedule there is no such V_ss, "
                        "and quoting a single stationary threshold would be "
                        "misleading. kc_QS is still shown because it does not "
                        "depend on V_ss, but it is not the threshold this "
                        "model obeys."))
            return info

    def event_rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self.detector.to_rows()


# =============================================================================
# WORKER THREAD
# =============================================================================

class SimulationWorker:
    """Runs the integration off the interface thread.

    Justified by measurement, not by caution: advance() has a median cost of
    2-3 ms but a rapid-slip tail of 51-182 ms (benchmark_engine.py). On the UI
    thread that tail is a visible multi-frame freeze, occurring exactly during
    the event the user is trying to watch.

    The worker only calls controller.advance_frame(); every mutation is
    already inside the controller's lock, so the interface can read snapshots
    at any time.
    """

    def __init__(self, controller: SimulationController,
                 hz: Optional[int] = None):
        self.c = controller
        self.hz = hz or controller.SCENE_HZ
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames = 0
        self.last_frame_wall = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="rsf-integrator")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        period = 1.0 / self.hz
        t_prev = time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            real_dt = t0 - t_prev
            t_prev = t0
            # Cap the catch-up so a long stall cannot make one frame jump a
            # huge amount of model time.
            real_dt = min(real_dt, 4.0 * period)
            try:
                self.c.advance_frame(real_dt)
            except Exception as exc:                            # noqa: BLE001
                with self.c._lock:
                    self.c.playing = False
                    self.c.error = (f"Worker thread error: "
                                    f"{type(exc).__name__}: {exc}")
            self.frames += 1
            self.last_frame_wall = time.perf_counter() - t0
            time.sleep(max(0.0, period - self.last_frame_wall))
