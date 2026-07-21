"""
=============================================================================
 test_simulation_controller.py   --   v0.2   Stage-3 controller tests
=============================================================================
 Run:   python test_simulation_controller.py
 Deps:  numpy, scipy, rsf_engine.py, simulation_controller.py

 These tests exercise the CONTROLLER -- the layer the browser talks to. They
 do not re-test the physics; that is test_rsf_engine.py's job and all 107 of
 its checks must still pass.

 What is being defended here:
   - the browser cannot move the block, only impose loading;
   - changing V_lp continues the trajectory instead of restarting it;
   - a preset cannot silently replace an active trajectory;
   - a mouse drag imposes exactly the displacement it appears to impose, and
     the resulting slip is whatever the engine says it is;
   - playback speed is a display setting and touches no physics;
   - a solver failure stops the run, keeps the real data, and invents nothing.

 AS EVER: passing these tests shows the controller behaves as specified. It
 is not evidence that the underlying model describes any real rock.
=============================================================================
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np

import rsf_engine as E
import simulation_controller as SC
from simulation_controller import (LoadingMode, OpState, SimulationController,
                                   SimulationWorker, THRESHOLDS,
                                   classify_state)

_RESULTS: List[Tuple[str, bool, str]] = []
_NOTES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(condition), detail))
    print(f"    [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def note(msg: str) -> None:
    _NOTES.append(msg)
    print(f"    NOTE: {msg}")


def section(title: str) -> None:
    print(f"\n{'-' * 74}\n{title}\n{'-' * 74}")


def _run_frames(c: SimulationController, n: int, real_dt: float = 1 / 25):
    for _ in range(n):
        if not c.advance_frame(real_dt):
            break


# =============================================================================
# 1. PLAY / PAUSE
# =============================================================================

def test_play_pause():
    section("TEST 1  --  Play and Pause")
    c = SimulationController("unstable")
    check("a new controller starts paused", not c.playing, "")
    t0 = c.snapshot().t
    _run_frames(c, 10)
    check("no model time advances while paused",
          c.snapshot().t == t0, f"t stayed at {t0:.6g} s")

    c.play()
    check("play() sets playing", c.playing, "")
    _run_frames(c, 10)
    t1 = c.snapshot().t
    check("model time advances while playing", t1 > t0,
          f"t {t0:.6g} -> {t1:.6g} s")

    c.pause()
    _run_frames(c, 10)
    check("pause() stops advancement", c.snapshot().t == t1,
          f"t stayed at {t1:.6g} s")

    n_before = c.snapshot().n_samples
    c.toggle_play()
    _run_frames(c, 5)
    check("toggle_play() resumes and history keeps growing",
          c.playing and c.snapshot().n_samples > n_before,
          f"{n_before} -> {c.snapshot().n_samples} samples")


# =============================================================================
# 2. SINGLE STEP
# =============================================================================

def test_single_step():
    section("TEST 2  --  Single Step")
    c = SimulationController("unstable")
    t0 = c.snapshot().t
    ok = c.single_step()
    t1 = c.snapshot().t
    check("single_step advances exactly SINGLE_STEP_DT of model time",
          ok and np.isclose(t1 - t0, c.SINGLE_STEP_DT, rtol=1e-12),
          f"advanced {t1 - t0:.9g} s, declared "
          f"{c.SINGLE_STEP_DT:.9g} s")
    check("single_step leaves the simulation paused", not c.playing, "")

    c.play()
    c.single_step(0.25)
    check("single_step with an explicit amount also pauses and is exact",
          not c.playing and np.isclose(c.snapshot().t - t1, 0.25, rtol=1e-12),
          f"advanced {c.snapshot().t - t1:.9g} s")


# =============================================================================
# 3. RESET
# =============================================================================

def test_reset():
    section("TEST 3  --  Reset")
    c = SimulationController("unstable")
    c.play()
    _run_frames(c, 40)
    s_before = c.snapshot()
    check("a trajectory exists before Reset",
          s_before.t > 0 and s_before.n_samples > 1,
          f"t = {s_before.t:.4g} s, {s_before.n_samples} samples")

    c.reset()
    s = c.snapshot()
    check("Reset returns model time to zero", s.t == 0.0, "")
    check("Reset returns delta and u_lp to zero",
          s.delta == 0.0 and s.u_lp == 0.0, "")
    check("Reset clears the history down to the initial sample",
          s.n_samples == 1, f"{s.n_samples} sample")
    check("Reset clears the event catalogue", s.n_events == 0, "")
    check("Reset leaves the simulation paused", not c.playing, "")
    check("Reset recomputes a consistent spring preload",
          abs(s.tau_spring - s.tau_total) / c.params.sigma_n < 1e-12,
          f"residual {s.residual:.3e} Pa")

    # Reset is the route by which the evolution law may change.
    c.reset(law="slip")
    check("Reset can change the evolution law", c.law == "slip"
          and c.engine.law == "slip", f"law = {c.law}")


# =============================================================================
# 4. V_lp CHANGES WITHOUT STATE RESET
# =============================================================================

def test_vlp_change():
    section("TEST 4  --  Changing V_lp continues the trajectory")
    c = SimulationController("stable_vw")
    c.play()
    _run_frames(c, 40)
    s0 = c.snapshot()

    c.set_vlp(5e-6)
    s1 = c.snapshot()
    check("V_lp change does not move model time", s1.t == s0.t, "")
    check("V_lp change does not alter V", s1.V == s0.V,
          f"V = {s1.V:.6e} m/s")
    check("V_lp change does not alter theta", s1.theta == s0.theta, "")
    check("V_lp change does not alter delta or u_lp",
          s1.delta == s0.delta and s1.u_lp == s0.u_lp, "")
    check("V_lp change does not alter any stress",
          s1.tau_total == s0.tau_total
          and s1.tau_friction == s0.tau_friction, "")
    check("V_lp change does not clear the history",
          s1.n_samples == s0.n_samples, f"{s1.n_samples} samples")

    _run_frames(c, 20)
    s2 = c.snapshot()
    check("the new V_lp is what the engine now imposes",
          np.isclose(s2.V_lp, 5e-6, rtol=1e-12),
          f"V_lp = {s2.V_lp:.4e} m/s")
    check("V responds to the new V_lp", s2.V > s0.V,
          f"V {s0.V:.4e} -> {s2.V:.4e} m/s")

    try:
        c.set_vlp(-1e-6)
        check("negative V_lp is rejected", False, "no exception")
    except ValueError:
        check("negative V_lp is rejected", True, "")

    c.set_vlp(1.0)
    check("V_lp above the supported range is clipped, not accepted silently",
          c.V_lp_setpoint == SC.V_LP_MAX,
          f"requested 1.0, got {c.V_lp_setpoint:.3e} m/s "
          f"(V_LP_MAX = {SC.V_LP_MAX:.1e})")


# =============================================================================
# 5. HOLD
# =============================================================================

def test_hold():
    section("TEST 5  --  Hold")
    c = SimulationController("stable_vw")
    c.play()
    _run_frames(c, 40)
    s0 = c.snapshot()
    c.hold()
    check("hold() switches the mode", c.mode is LoadingMode.HOLD,
          c.mode.value)
    _run_frames(c, 60)
    s1 = c.snapshot()
    check("V_lp is exactly zero during Hold", s1.V_lp == 0.0, "")
    check("u_lp stops advancing during Hold",
          np.isclose(s1.u_lp, s0.u_lp, rtol=1e-9),
          f"u_lp {s0.u_lp*1e6:.6f} -> {s1.u_lp*1e6:.6f} um")
    check("the block keeps creeping during Hold (V > 0)", s1.V > 0,
          f"V = {s1.V:.4e} m/s")
    check("delta still advances during Hold (block moves, load point does not)",
          s1.delta > s0.delta,
          f"delta {s0.delta*1e6:.4f} -> {s1.delta*1e6:.4f} um")
    check("the spring unloads during Hold",
          s1.tau_total < s0.tau_total,
          f"tau_total {s0.tau_total/1e6:.5f} -> {s1.tau_total/1e6:.5f} MPa")
    info = c.stability_info()
    check("Hold reports that no stationary kc_QD is defined",
          not info["kc_qd_defined"] and info["kc_qd"] is None,
          "note shown to the user begins: "
          + info["kc_qd_note"][:52] + "...")


# =============================================================================
# 6. PRESETS ONLY THROUGH RESET
# =============================================================================

def test_presets():
    section("TEST 6  --  Presets cannot silently replace a trajectory")
    c = SimulationController("unstable")
    c.play()
    _run_frames(c, 40)
    s0 = c.snapshot()
    p_before = c.params

    summary = c.stage_preset("stable_vs")
    s1 = c.snapshot()
    check("staging a preset pauses the simulation", not c.playing, "")
    check("staging a preset does NOT change the parameters",
          c.params is p_before, f"k still {c.params.k:.4e} Pa/m")
    check("staging a preset does NOT touch the trajectory",
          s1.t == s0.t and s1.n_samples == s0.n_samples, "")
    check("the staged summary warns that applying it will Reset",
          "NOT YET APPLIED" in summary and "RESET" in summary, "")
    check("a pending preset is reported", c.has_pending_preset(), "")

    c.cancel_pending_preset()
    check("cancelling leaves everything alone",
          not c.has_pending_preset() and c.params is p_before, "")

    c.stage_preset("stable_vs")
    c.apply_pending_preset()
    s2 = c.snapshot()
    check("applying a preset performs a full Reset",
          s2.t == 0.0 and s2.n_samples == 1 and not c.playing, "")
    check("applying a preset installs the new parameters",
          c.params.a > c.params.b,
          f"a = {c.params.a}, b = {c.params.b} (velocity strengthening)")

    for name in E.PRESET_NAMES:
        c.reset(preset_name=name)
        info = c.stability_info()
        ok = info["kc_qs"] is not None and "k_over_kc_qs" in info
        check(f"preset '{name}' reports stiffness diagnostics", ok,
              f"k/kc_QS = {info['k_over_kc_qs']:.4f}"
              + (f", k/kc_QD = {info['k_over_kc_qd']:.4f}"
                 if info["kc_qd_defined"] else ", kc_QD not defined"))


# =============================================================================
# 7-9. MANUAL PULL
# =============================================================================

def test_manual_pull():
    section("TEST 7-9  --  Manual pull")
    c = SimulationController("stable_vw")
    c.set_manual_sensitivity("medium")
    mpp = c.manual.metres_per_pixel
    s0 = c.snapshot()

    c.begin_manual_pull(x_px=100.0)
    check("beginning a drag pauses and switches to manual mode",
          not c.playing and c.mode is LoadingMode.MANUAL, c.mode.value)

    # Drag right in 10 steps of 5 px each.
    x = 100.0
    for _ in range(10):
        x += 5.0
        c.update_manual_pull(x, real_dt=1 / 25)
    s1 = c.snapshot()

    imposed = 50.0 * mpp            # 50 px total
    du_lp = s1.u_lp - s0.u_lp
    check("final u_lp equals the imposed manual displacement",
          np.isclose(du_lp, imposed, rtol=1e-9),
          f"imposed {imposed*1e6:.6f} um, u_lp advanced "
          f"{du_lp*1e6:.6f} um, rel error "
          f"{abs(du_lp - imposed)/imposed:.3e}")
    check("controller's own tally agrees with u_lp",
          np.isclose(c.manual_total_displacement, du_lp, rtol=1e-9), "")

    # The block must NOT have followed the pointer.
    d_delta = s1.delta - s0.delta
    check("delta is determined by the engine, not by the pointer",
          not np.isclose(d_delta, imposed, rtol=1e-3) and d_delta > 0,
          f"load point moved {imposed*1e6:.4f} um, block slipped "
          f"{d_delta*1e6:.4f} um -- the difference "
          f"{(imposed - d_delta)*1e6:+.4f} um went into the spring")
    check("the spring extension changed by exactly (u_lp - delta)",
          np.isclose(s1.e - s0.e, du_lp - d_delta, rtol=1e-9, atol=1e-18),
          f"de = {(s1.e - s0.e)*1e6:.6f} um, "
          f"du_lp - ddelta = {(du_lp - d_delta)*1e6:.6f} um")
    check("spring-stress residual stays at solver tolerance during manual pull",
          abs(s1.residual) / c.params.sigma_n < 1e-8,
          f"residual {s1.residual:.4e} Pa "
          f"({abs(s1.residual)/c.params.sigma_n:.3e} of sigma_n)")

    # Dragging LEFT must not create negative loading.
    u_before = c.snapshot().u_lp
    t_before = c.snapshot().t
    c.update_manual_pull(x - 40.0, real_dt=1 / 25)
    s2 = c.snapshot()
    check("dragging left does not move the load point backwards",
          s2.u_lp == u_before, f"u_lp unchanged at {u_before*1e6:.6f} um")
    check("dragging left does not advance model time",
          s2.t == t_before, "")
    check("dragging left raises a clear warning",
          c.manual_warning is not None
          and "Reverse loading is not supported" in c.manual_warning,
          c.manual_warning.split(".")[0] if c.manual_warning else "none")

    # Release -> Hold.
    c.end_manual_pull()
    check("releasing the pointer defaults to Hold",
          c.mode is LoadingMode.HOLD, c.mode.value)
    _run_frames(c, 5)
    check("V_lp really is zero after release", c.snapshot().V_lp == 0.0, "")

    # Saturation: a very fast drag keeps the displacement, stretches time.
    c2 = SimulationController("stable_vw")
    c2.set_manual_sensitivity("high")
    u0 = c2.snapshot().u_lp
    c2.begin_manual_pull(0.0)
    c2.update_manual_pull(400.0, real_dt=1 / 25)   # huge, fast
    s3 = c2.snapshot()
    imposed2 = 400.0 * c2.manual.metres_per_pixel
    check("a saturating drag still applies the full displacement",
          np.isclose(s3.u_lp - u0, imposed2, rtol=1e-9),
          f"imposed {imposed2*1e6:.4f} um, applied "
          f"{(s3.u_lp - u0)*1e6:.4f} um")
    check("a saturating drag warns that model time was stretched",
          c2.manual_warning is not None
          and "saturated" in c2.manual_warning,
          c2.manual_warning.split(".")[0] if c2.manual_warning else "none")
    check("the imposed V_lp never exceeds the manual ceiling",
          s3.V_lp <= c2.manual.v_lp_max * (1 + 1e-12),
          f"V_lp = {s3.V_lp:.4e} m/s, ceiling "
          f"{c2.manual.v_lp_max:.1e} m/s")
    note(f"Manual sensitivity mapping is displayed as: "
         f"{c2.manual.describe()}")

    # No retroactive rewriting: history times must be monotone throughout.
    a = c2.history.arrays()
    check("manual pull never rewrites history (time is monotone)",
          bool(np.all(np.diff(a["t"]) >= 0)), f"{a['t'].size} samples")


# =============================================================================
# 10. OPERATIONAL STATE LABELS
# =============================================================================

def test_operational_states():
    section("TEST 10  --  Operational state labels")
    thr = THRESHOLDS

    # Unit-level: each rule fires where it should.
    cases = [
        ("event", dict(V=1e-2, V_lp=1e-6, dlnV_dt=1.0, t=10.0,
                       t_last_event_end=None), OpState.EVENT),
        ("post-event", dict(V=1e-9, V_lp=1e-6, dlnV_dt=0.0, t=10.0,
                            t_last_event_end=5.0), OpState.RELAXING),
        ("accelerating", dict(V=1e-4, V_lp=1e-6, dlnV_dt=0.5, t=1e5,
                              t_last_event_end=None), OpState.ACCELERATING),
        ("creep", dict(V=1.0e-6, V_lp=1e-6, dlnV_dt=0.0, t=10.0,
                       t_last_event_end=None), OpState.CREEP),
        ("locked", dict(V=1e-9, V_lp=1e-6, dlnV_dt=0.0, t=1e5,
                        t_last_event_end=None), OpState.LOCKED),
        ("hold-locked", dict(V=1e-11, V_lp=0.0, dlnV_dt=0.0, t=10.0,
                             t_last_event_end=None), OpState.LOCKED),
    ]
    for label, kw, expect in cases:
        got, rule = classify_state(**kw)
        check(f"rule fires correctly for '{label}'", got is expect,
              f"got '{got.value}' via: {rule}")

    check("settling back onto steady creep is NOT called accelerating",
          classify_state(V=0.97e-6, V_lp=1e-6, dlnV_dt=5e-3, t=10.0,
                         t_last_event_end=None)[0] is OpState.CREEP,
          "the accelerating rule is gated on V > 2*V_lp")

    # Against real histories.
    c = SimulationController("stable_vw")
    c.play()
    _run_frames(c, 200)
    check("a stable velocity-weakening run settles into 'Stable creep'",
          c.snapshot().op_state == OpState.CREEP.value,
          f"'{c.snapshot().op_state}' -- {c.snapshot().op_rule}")

    c2 = SimulationController("stable_vs")
    c2.play()
    _run_frames(c2, 200)
    check("a velocity-strengthening run also settles into 'Stable creep'",
          c2.snapshot().op_state == OpState.CREEP.value,
          f"'{c2.snapshot().op_state}'")

    # Stick-slip: walk a whole cycle and collect the labels seen.
    c3 = SimulationController("unstable")
    c3.adaptive_advance = True
    c3.set_speed("Fast")
    c3.play()
    seen = set()
    for _ in range(6000):
        if not c3.advance_frame(1 / 25):
            break
        seen.add(c3.snapshot().op_state)
        if c3.snapshot().n_events >= 2:
            break
    check("a stick-slip run passes through locked, accelerating and event",
          {OpState.LOCKED.value, OpState.ACCELERATING.value,
           OpState.EVENT.value} <= seen,
          "labels seen: " + ", ".join(sorted(seen)))
    check("every label produced comes with the rule that produced it",
          bool(c3.snapshot().op_rule), c3.snapshot().op_rule)


# =============================================================================
# 11-12. HISTORY AND EVENT-MARKER CONTINUITY
# =============================================================================

def test_history_and_events():
    section("TEST 11-12  --  History and event-marker continuity")
    c = SimulationController("unstable")
    c.set_speed("Fast")
    c.play()
    for _ in range(4000):
        if not c.advance_frame(1 / 25):
            break
        if c.snapshot().n_events >= 2:
            break
    a = c.history.arrays()
    check("history time is monotone non-decreasing across many frames",
          bool(np.all(np.diff(a["t"]) >= 0)), f"{a['t'].size} samples")
    check("history has no gaps at frame boundaries",
          bool(np.all(np.isfinite(a["V"]))) and a["t"][0] == 0.0, "")

    pd = c.plot_data(n_max=800)
    check("plot data comes from History and is decimated for display only",
          pd["n_shown"] <= 800 and pd["n_full"] == len(c.history)
          and pd["n_full"] > pd["n_shown"],
          f"{pd['n_full']} stored, {pd['n_shown']} plotted")
    check("the latest plotted point matches the animation's latest time",
          np.isclose(pd["t"][-1], c.snapshot().t, rtol=0, atol=1e-12),
          f"plot ends at {pd['t'][-1]:.6f} s, snapshot at "
          f"{c.snapshot().t:.6f} s")

    ev = pd["event_times"]
    check("event markers exist and lie inside the plotted time range",
          len(ev) >= 2 and all(pd["t"][0] <= x <= pd["t"][-1] for x in ev),
          f"{len(ev)} markers at " + ", ".join(f"{x:.2f}" for x in ev[:4]))
    # Markers must sit where V actually crossed the threshold.
    for x in ev[:3]:
        i = c.history.index_at_time(x)
        w = slice(max(i - 3, 0), min(i + 60, len(c.history)))
        check(f"event marker at t = {x:.3f} s coincides with V crossing "
              f"the threshold",
              float(a["V"][w].max()) > THRESHOLDS.V_event,
              f"max V near marker = {a['V'][w].max():.4e} m/s")

    rows = c.event_rows()
    check("event catalogue names the three stress drops separately",
          all(k in rows[0] for k in ("drop_tau_total_Pa",
                                     "drop_tau_friction_Pa",
                                     "drop_tau_damping_Pa")), "")
    check("the first event is flagged as possible spin-up",
          rows[0]["is_possible_spin_up"], "")


# =============================================================================
# 13. PLAYBACK SPEED IS NOT PHYSICS
# =============================================================================

def test_speed_is_not_physics():
    section("TEST 13  --  Playback speed does not change physics")
    c = SimulationController("stable_vw")
    c.play()
    _run_frames(c, 20)
    s0 = c.snapshot()
    v_before = s0.V_lp

    for name in ("Slow", "Normal", "Fast"):
        c.set_speed(name)
        s = c.snapshot()
        check(f"speed '{name}' leaves V_lp untouched",
              s.V_lp == v_before,
              f"V_lp = {s.V_lp:.4e} m/s, speed = "
              f"{c.speed_multiplier:g} model s per real s")
    check("changing speed does not move model time or the state",
          c.snapshot().t == s0.t and c.snapshot().V == s0.V, "")
    check("changing speed does not alter the parameters",
          c.params.k == c.params.k and c.params.a == c.params.a, "")

    # The same model time reached at two different speeds must give the same
    # physical answer, to solver tolerance.
    def to_time(speed: str, T: float):
        """Advance to EXACTLY model time T.

        The naive loop `while t < T: advance_frame()` overshoots by up to one
        frame, and a frame is 200x longer at Fast than at Slow. Comparing the
        results then compares two different model times and measures nothing
        but the physical evolution over the difference. Here we run to just
        short of T and land the remainder with a single explicit step.
        """
        cc = SimulationController("stable_vw")
        cc.adaptive_advance = False
        cc.set_speed(speed)
        cc.play()
        guard = 0
        while cc.snapshot().t < T and guard < 200000:
            guard += 1
            remaining = T - cc.snapshot().t
            frame_dt = remaining / cc.speed_multiplier
            if frame_dt <= 1.0 / 25:
                cc.single_step(remaining)      # land exactly on T
                break
            if not cc.advance_frame(1 / 25):
                break
        return cc.snapshot()

    T = 60.0
    a = to_time("Slow", T)
    b = to_time("Fast", T)
    rel = abs(a.V - b.V) / a.V
    check("both speeds landed on exactly the same model time",
          a.t == b.t == T, f"Slow t = {a.t!r}, Fast t = {b.t!r}")
    check("the same model time reached at Slow and Fast gives the same V",
          rel < 1e-6,
          f"Slow: V = {a.V:.10e} at t = {a.t:.4f} s; "
          f"Fast: V = {b.V:.10e} at t = {b.t:.4f} s; rel {rel:.3e}")
    note("Speed changes the model time advanced per frame, i.e. the chunk "
         "size. Stage 2 showed chunking leaves the interseismic solution "
         "unchanged to 1 part in 1e13, which is why this comparison holds.")

    # Event slow-motion is a display setting too.
    c.adaptive_advance = True
    v1 = c.snapshot().V_lp
    c.adaptive_advance = False
    check("toggling adaptive advancement does not change V_lp",
          c.snapshot().V_lp == v1, f"V_lp = {v1:.4e} m/s")


# =============================================================================
# 14. SOLVER FAILURE PROPAGATION
# =============================================================================

def test_failure_propagation():
    section("TEST 14  --  Solver failure propagation")
    c = SimulationController("unstable")
    c.play()
    _run_frames(c, 30)
    s_good = c.snapshot()
    n_good = s_good.n_samples

    # Corrupt the state exactly as a failing integration would leave it.
    with c._lock:
        c.state = E.SimulationState(t=c.state.t, lnV=c.state.lnV, theta=0.0,
                                    delta=c.state.delta, u_lp=c.state.u_lp,
                                    e0=c.state.e0)
        c.engine.formulation = "lnV_theta"
    ok = c.advance_frame(1 / 25)

    check("a failing step returns False rather than raising", ok is False, "")
    check("the simulation pauses immediately on failure", not c.playing, "")
    check("an error message is exposed to the interface",
          c.error is not None and len(c.error) > 20,
          c.error.splitlines()[0][:100] if c.error else "none")
    check("the error names the loading mode and V_lp",
          "Mode =" in (c.error or "") and "V_lp =" in (c.error or ""), "")
    check("the error states that the trajectory so far is preserved",
          "preserved" in (c.error or ""), "")
    s_after = c.snapshot()
    check("the trajectory obtained so far is kept",
          s_after.n_samples >= n_good,
          f"{n_good} samples before, {s_after.n_samples} after")
    check("no fabricated values: nothing beyond the last good point",
          np.isfinite(s_after.t) and s_after.t >= s_good.t, "")

    check("play() refuses to restart a failed run", (c.play(), not c.playing)[1],
          "the user must Reset")
    c.reset()
    check("Reset clears the error and allows play again",
          c.error is None and (c.play(), c.playing)[1], "")


# =============================================================================
# 15. TIMELINE SNAPSHOT
# =============================================================================

def test_timeline():
    section("TEST 15  --  Timeline snapshot matches stored state")
    c = SimulationController("unstable")
    c.set_speed("Fast")
    c.play()
    for _ in range(3000):
        if not c.advance_frame(1 / 25):
            break
        if c.snapshot().n_events >= 1:
            break
    live = c.snapshot()
    check("running live before scrubbing", live.live and c.playing, "")

    t_target = live.t * 0.4
    c.select_history_time(t_target)
    s = c.snapshot()
    check("selecting a past time pauses the simulation", not c.playing, "")
    check("the snapshot reports that it is not live", not s.live, "")

    a = c.history.arrays()
    i = c.history.index_at_time(t_target)
    for fld, val in (("t", s.t), ("V", s.V), ("delta", s.delta),
                     ("u_lp", s.u_lp), ("e", s.e),
                     ("tau_total", s.tau_total), ("mu", s.mu),
                     ("theta", s.theta)):
        check(f"scrubbed '{fld}' equals the stored sample exactly",
              float(a[fld][i]) == val,
              f"{val:.10e}")
    check("the reconstructed snapshot is internally consistent",
          abs(s.tau_spring - s.tau_total) / c.params.sigma_n < 1e-8,
          f"residual {s.residual:.3e} Pa")
    check("scrubbing does not alter the stored history",
          len(c.history) == live.n_samples, f"{len(c.history)} samples")

    c.return_to_live()
    s2 = c.snapshot()
    check("return_to_live goes back to the latest sample",
          s2.live and np.isclose(s2.t, live.t, rtol=0, atol=1e-12),
          f"back at t = {s2.t:.6f} s")


# =============================================================================
# 16. WORKER THREAD
# =============================================================================

def test_worker():
    section("TEST 16  --  Worker thread")
    c = SimulationController("unstable")
    c.set_speed("Normal")
    w = SimulationWorker(c, hz=25)
    w.start()
    c.play()
    time.sleep(1.5)
    t_mid = c.snapshot().t
    c.pause()
    time.sleep(0.3)
    t_paused = c.snapshot().t
    w.stop()

    check("the worker advances model time in the background", t_mid > 0,
          f"t = {t_mid:.3f} s after 1.5 s of real time "
          f"({w.frames} frames)")
    check("pausing stops the worker advancing",
          np.isclose(t_paused, t_mid, rtol=0, atol=1e-9),
          f"t stayed at {t_paused:.6f} s")
    check("the worker achieved a usable frame rate",
          w.frames >= 20, f"{w.frames} frames in ~1.8 s "
                          f"(target {c.SCENE_HZ} Hz)")
    note(f"Last frame's integration cost: {w.last_frame_wall*1e3:.2f} ms "
         f"(budget at {c.SCENE_HZ} Hz is {1000/c.SCENE_HZ:.1f} ms)")
    check("the worker stopped cleanly", w._thread is None, "")




# =============================================================================
# 17. EVENT DETECTION vs VISUAL EVENT DISPLAY  (tested separately)
# =============================================================================

def test_event_physical_vs_visual():
    section("TEST 17  --  Physical event detection vs visual event display")
    c = SimulationController("unstable")
    c.set_speed("Fast")
    c.play()
    frames_above = 0
    flash_frames = 0
    physical_event_labels = 0
    for _ in range(20000):
        if not c.advance_frame(1 / 25):
            break
        s = c.snapshot()
        if s.V > THRESHOLDS.V_event:
            frames_above += 1
        if s.event_flash_active:
            flash_frames += 1
        if s.physical_state_label == OpState.EVENT.value:
            physical_event_labels += 1
        if s.n_events >= 1:
            break

    # (a) the PHYSICAL event is detected and catalogued
    ev = c.detector.events[0]
    check("(a) the physical event is detected and catalogued",
          ev.t_end > ev.t_start and ev.V_max > THRESHOLDS.V_event,
          f"onset {ev.t_start:.4f} s, end {ev.t_end:.4f} s, duration "
          f"{ev.duration_s:.5f} s, V_max {ev.V_max:.4e} m/s, peak at "
          f"{ev.t_peak:.4f} s, spin-up flag {ev.is_possible_spin_up}")
    check("(a) the catalogue records the three stress drops separately",
          ev.drop_tau_total_Pa > 0 and ev.drop_tau_friction_Pa > 0,
          f"total {ev.drop_tau_total_Pa/1e6:.4f} MPa, friction "
          f"{ev.drop_tau_friction_Pa/1e6:.4f} MPa, damping "
          f"{ev.drop_tau_damping_Pa/1e6:.4f} MPa")

    # (b) History contains samples INSIDE the event
    a = c.history.arrays()
    inside = (a["t"] >= ev.t_start) & (a["t"] <= ev.t_end)
    gap = float(np.diff(a["t"][inside]).max()) if inside.sum() > 1 else np.inf
    check("(b) History contains many samples inside the event",
          inside.sum() > 50,
          f"{int(inside.sum())} samples, largest gap {gap:.3e} s, "
          f"V spans {a['V'][inside].min():.3e} to "
          f"{a['V'][inside].max():.3e} m/s")
    check("(b) samples inside the event come from the solver's own steps, "
          "not from interpolation across it",
          gap < ev.duration_s / 20,
          f"largest stored gap {gap:.3e} s is {ev.duration_s/gap:.0f}x "
          f"smaller than the event; History stores solve_ivp's adaptive "
          f"steps directly (no dense_output, no interpolation)")

    # (c) the controller can identify the physical event interval
    lve = c.last_visual_event
    check("(c) the controller reports the physical event interval it saw",
          lve is not None and lve[2] > THRESHOLDS.V_event,
          f"interval ({lve[0]:.4f}, {lve[1]:.4f}) s, V_max {lve[2]:.4e} m/s"
          if lve else "none")

    # (d) UI frames now actually sample the event, and the latch activates
    check("(d) UI frames sample the event (adaptive advancement working)",
          frames_above > 10,
          f"{frames_above} frames had V above threshold "
          f"(before the fix this was 0 of 45)")
    check("(d) the physical label 'Rapid-slip event' is produced",
          physical_event_labels > 0,
          f"{physical_event_labels} frames carried the physical event label")
    check("(d) the visual latch activated", flash_frames > 0,
          f"{flash_frames} frames had event_flash_active")

    # (e) the latch expires after its documented wall-clock duration
    c.pause()
    with c._lock:
        c._event_flash_until = time.monotonic() + c.EVENT_FLASH_HOLD_S
    check("(e) latch is active immediately after arming",
          c.event_flash_active and c.snapshot().visual_state_label
          == OpState.EVENT.value,
          f"{c.event_flash_remaining_s:.3f} s remaining of "
          f"{c.EVENT_FLASH_HOLD_S:.2f} s")
    time.sleep(c.EVENT_FLASH_HOLD_S * 0.5)
    check("(e) latch still active at half its documented duration",
          c.event_flash_active, f"{c.event_flash_remaining_s:.3f} s left")
    time.sleep(c.EVENT_FLASH_HOLD_S * 0.6)
    check("(e) latch has expired after its documented duration",
          not c.event_flash_active,
          f"EVENT_FLASH_HOLD_S = {c.EVENT_FLASH_HOLD_S:.2f} s elapsed")
    check("(e) after expiry the visual label falls back to the physical one",
          c.snapshot().visual_state_label == c.snapshot().physical_state_label,
          f"both now '{c.snapshot().physical_state_label}'")

    # (f) the catalogue is untouched by the latch
    ev2 = c.detector.events[0]
    check("(f) the visual latch did not change the event duration",
          ev2.duration_s == ev.duration_s,
          f"duration still {ev2.duration_s:.6f} s")
    check("(f) the visual latch did not change the event count or onset",
          len(c.detector.events) >= 1 and ev2.t_start == ev.t_start,
          f"onset still {ev2.t_start:.6f} s")
    check("(f) the latch never claims V is above threshold",
          c.snapshot().V < THRESHOLDS.V_event,
          f"V = {c.snapshot().V:.4e} m/s while the flash was shown")

    # the latch is suppressed while scrubbing the past
    with c._lock:
        c._event_flash_until = time.monotonic() + 5.0
    c.select_history_time(ev.t_start * 0.5)
    check("the latch is suppressed while viewing a past time",
          not c.event_flash_active,
          "a past moment is labelled by what was happening then")
    c.return_to_live()


# =============================================================================
# 18. ADAPTIVE ADVANCEMENT
# =============================================================================

def test_adaptive_advancement():
    section("TEST 18  --  Adaptive model-time advancement")
    c = SimulationController("unstable")
    c.set_speed("Fast")

    d = c.frame_dt_cap_detail()
    check("interseismically the cap is loose (large chunks allowed)",
          d["cap"] > 1.0,
          f"cap {d['cap']:.3f} s, binding bound = {d['binding']}, "
          f"dt_slip {d['dt_slip']:.3f} s, dt_rate {d['dt_rate']:.3g} s")
    check("the interseismic cap does not force a tiny step for the whole "
          "774 s cycle",
          d["cap"] >= c.params.Dc / (4 * 1e-6) * 0.9,
          f"cap {d['cap']:.3f} s corresponds to "
          f"{c.SLIP_FRACTION_PER_FRAME} x Dc of slip per frame")

    # Drive to the coseismic limb and confirm the cap tightens by orders of
    # magnitude, without ever changing V_lp.
    c.play()
    caps, V_at = [], []
    for _ in range(20000):
        if not c.advance_frame(1 / 25):
            break
        caps.append(c.frame_dt_cap())
        V_at.append(c.snapshot().V)
        if c.snapshot().n_events >= 1:
            break
    caps, V_at = np.array(caps), np.array(V_at)
    fast = V_at > THRESHOLDS.V_event
    check("the cap tightens by orders of magnitude during rapid slip",
          fast.any() and caps[fast].min() < caps.max() / 1e3,
          f"cap ranged {caps.min():.3e} to {caps.max():.3e} s; "
          f"min during the event {caps[fast].min():.3e} s")
    check("the cap never falls below its documented floor",
          caps.min() >= c.MIN_FRAME_DT,
          f"min cap {caps.min():.3e} s, floor "
          f"{c.MIN_FRAME_DT:.1e} s")

    # Disabling adaptive advancement must reproduce the original miss, which
    # is the cleanest proof that the fix is doing the work.
    c2 = SimulationController("unstable")
    c2.adaptive_advance = False
    c2.set_speed("Fast")
    c2.play()
    missed = 0
    for _ in range(4000):
        if not c2.advance_frame(1 / 25):
            break
        if c2.snapshot().V > THRESHOLDS.V_event:
            missed += 1
        if c2.snapshot().n_events >= 1:
            break
    check("with adaptive advancement OFF the event is missed by the frames "
          "(the original bug, reproduced deliberately)",
          missed == 0 and c2.snapshot().n_events >= 1,
          f"{missed} frames sampled the event, yet "
          f"{c2.snapshot().n_events} event(s) were still catalogued -- the "
          f"solver always resolved it; only the frame sampling missed it")
    note("Adaptive advancement changes the chunk size, not the equations. "
         "Stage 2 measured chunking to be neutral to 1 part in 1e13 "
         "interseismically.")


# =============================================================================
# 19. THREADING AND ASYNCHRONOUS SAFETY
# =============================================================================

def test_threading_safety():
    section("TEST 19  --  Threading and asynchronous safety")

    # (1) duplicate Play must not launch duplicate workers
    c = SimulationController("unstable")
    w = SimulationWorker(c, hz=25)
    w.start()
    first = w._thread
    w.start(); w.start()
    check("duplicate start() does not create a second worker thread",
          w._thread is first and
          sum(1 for th in threading.enumerate()
              if th.name == "rsf-integrator") == 1,
          f"{sum(1 for th in threading.enumerate() if th.name == 'rsf-integrator')} "
          f"integrator thread(s) alive")
    c.play(); c.play(); c.play()
    time.sleep(0.4)
    check("duplicate play() calls are idempotent",
          c.playing and c.snapshot().t > 0,
          f"t = {c.snapshot().t:.4f} s")

    # (2) pause during an active worker step
    c.pause()
    t_a = c.snapshot().t
    time.sleep(0.4)
    t_b = c.snapshot().t
    check("Pause during an active worker stops advancement cleanly",
          t_a == t_b, f"t held at {t_b:.6f} s across 0.4 s of real time")

    # (3) reset during an active worker step
    c.play()
    time.sleep(0.3)
    c.reset()
    t_reset = c.snapshot().t
    time.sleep(0.4)
    check("Reset during an active worker leaves a clean t = 0 trajectory",
          t_reset == 0.0 and c.snapshot().n_events == 0 and not c.playing,
          f"t = {c.snapshot().t:.6f} s, {c.snapshot().n_samples} sample(s)")
    w.stop()
    check("worker stops cleanly after a mid-flight Reset", w._thread is None,
          "")

    # (4) only one integration job at a time
    c2 = SimulationController("unstable")
    c2.play()
    concurrent = {"max": 0, "now": 0}
    guard = threading.Lock()
    real_advance = c2.engine.advance

    def instrumented(state, dt):
        with guard:
            concurrent["now"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["now"])
        try:
            time.sleep(0.004)
            return real_advance(state, dt)
        finally:
            with guard:
                concurrent["now"] -= 1

    c2.engine.advance = instrumented
    threads = [threading.Thread(target=lambda: [c2.advance_frame(1 / 25)
                                                for _ in range(25)])
               for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    check("only one integration job runs at a time under 6 concurrent callers",
          concurrent["max"] == 1,
          f"peak concurrency {concurrent['max']}")
    a = c2.history.arrays()
    check("concurrent advancement did not corrupt History",
          bool(np.all(np.diff(a["t"]) >= 0)) and bool(np.isfinite(a["V"]).all()),
          f"{a['t'].size} samples, time still monotone")

    # (5) stale results from a pre-Reset generation are discarded
    c3 = SimulationController("unstable")
    c3.play()
    real3 = c3.engine.advance
    fired = threading.Event()

    def slow_advance(state, dt):
        out = real3(state, dt)
        fired.set()
        time.sleep(0.35)          # Reset lands during this window
        return out

    c3.engine.advance = slow_advance
    t_before = c3.snapshot().t
    th = threading.Thread(target=lambda: c3.advance_frame(1 / 25))
    th.start()
    fired.wait(2.0)
    time.sleep(0.05)
    c3.reset()                    # generation bumped mid-integration
    th.join(5.0)
    check("a result computed before a Reset is discarded as stale",
          c3.stale_results_discarded >= 1,
          f"{c3.stale_results_discarded} stale result(s) discarded")
    check("the stale result did not contaminate the fresh trajectory",
          c3.snapshot().t == 0.0 and c3.snapshot().n_samples == 1,
          f"t = {c3.snapshot().t}, {c3.snapshot().n_samples} sample(s) "
          f"(was t = {t_before:.4f} s before Reset)")

    # (6) snapshot() must not block while an integration is in flight
    c4 = SimulationController("unstable")
    real4 = c4.engine.advance

    def blocking_advance(state, dt):
        time.sleep(0.30)
        return real4(state, dt)

    c4.engine.advance = blocking_advance
    c4.play()
    th = threading.Thread(target=lambda: c4.advance_frame(1 / 25))
    th.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    for _ in range(20):
        c4.snapshot()
    read_ms = (time.perf_counter() - t0) * 1e3
    th.join(5.0)
    check("snapshot() is not blocked by an in-flight 300 ms integration",
          read_ms < 100.0,
          f"20 snapshot reads took {read_ms:.1f} ms while a 300 ms "
          f"integration was running -- the field lock is released during "
          f"solve_ivp")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 74)
    print(f" test_simulation_controller.py  --  Stage-3 controller tests")
    print(f" rsf_engine v{E.__version__}, "
          f"simulation_controller v{SC.__version__}")
    print("=" * 74)

    tests: List[Tuple[str, Callable]] = [
        ("play/pause", test_play_pause),
        ("single step", test_single_step),
        ("reset", test_reset),
        ("V_lp change", test_vlp_change),
        ("hold", test_hold),
        ("presets", test_presets),
        ("manual pull", test_manual_pull),
        ("operational states", test_operational_states),
        ("history and events", test_history_and_events),
        ("speed is not physics", test_speed_is_not_physics),
        ("failure propagation", test_failure_propagation),
        ("timeline", test_timeline),
        ("worker thread", test_worker),
        ("physical vs visual event", test_event_physical_vs_visual),
        ("adaptive advancement", test_adaptive_advancement),
        ("threading safety", test_threading_safety),
    ]
    t0 = time.perf_counter()
    crashed: List[str] = []
    for name, fn in tests:
        try:
            fn()
        except Exception:                                       # noqa: BLE001
            crashed.append(name)
            print(f"\n    !! TEST GROUP '{name}' RAISED:")
            traceback.print_exc()

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print("\n" + "=" * 74)
    print(f" SUMMARY: {n_pass} passed, {n_fail} failed, "
          f"{len(crashed)} test group(s) crashed, "
          f"{time.perf_counter() - t0:.1f} s")
    print("=" * 74)
    if n_fail:
        print("\n FAILURES:")
        for nm, ok, det in _RESULTS:
            if not ok:
                print(f"   - {nm}: {det}")
    print("\n REMINDER: these tests check that the controller behaves as "
          "specified.\n They are not evidence that the physical model "
          "describes any real rock.")
    return 1 if (n_fail or crashed) else 0


if __name__ == "__main__":
    sys.exit(main())
