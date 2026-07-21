"""
benchmark_engine.py -- Stage-3 preparatory measurement.

Measures the WALL-CLOCK cost of SpringSliderEngine.advance() for a range of
model-time chunk sizes in three dynamical regimes. The animation loop is
designed from these numbers, not from a guess.

The question the benchmark must answer: given a target frame interval (say
1/30 s of real time), how much MODEL time can we afford to advance per frame,
and does the integration need its own thread so the browser event loop is
never blocked?

Run:  python benchmark_engine.py
"""

from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

import numpy as np

import rsf_engine as E
from rsf_engine import ConstantLoading, RSFParams, SpringSliderEngine

BASE = dict(mu0=0.60, Dc=20e-6, V0=1e-6, sigma_n=10e6, G=30e9, cs=3000.0)
CHUNKS = (0.01, 0.1, 1.0, 5.0)
V_LP = 1e-6


def _params(ratio: float) -> RSFParams:
    return RSFParams.from_stiffness_ratio(ratio, "qd", V_ss=V_LP,
                                          a=0.005, b=0.010, **BASE)


def _time_advance(eng, state, dt, n_reps: int) -> Tuple[List[float], object]:
    """Advance n_reps times, recording the wall time of each call."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        state, samples, rep = eng.advance(state, dt)
        times.append(time.perf_counter() - t0)
        if not rep.success:
            break
    return times, state


def _advance_to_condition(eng, state, predicate, dt=1.0, max_steps=20000):
    """Step until predicate(V) is true. Used to place the state inside an
    interseismic lull or on the rising limb of an instability."""
    for _ in range(max_steps):
        state, samples, rep = eng.advance(state, dt)
        if not rep.success:
            raise RuntimeError(rep.message)
        if samples["V"].size and predicate(float(samples["V"][-1])):
            return state
    raise RuntimeError("condition never reached")


def bench_regime(label: str, eng, state0, n_reps: int) -> Dict[float, Dict]:
    print(f"\n  {label}")
    print(f"    {'chunk/s':>9}{'median ms':>12}{'mean ms':>10}{'max ms':>10}"
          f"{'model/real':>12}{'reps':>6}")
    out = {}
    for dt in CHUNKS:
        # Fresh copy of the starting state for every chunk size, so the
        # regimes are compared at the same point on the trajectory.
        st = E.SimulationState(*state0.as_tuple())
        times, _ = _time_advance(eng, st, dt, n_reps)
        med = statistics.median(times)
        out[dt] = dict(median=med, mean=statistics.mean(times),
                       max=max(times), ratio=dt / med, n=len(times))
        print(f"    {dt:>9.2f}{med*1e3:>12.3f}{statistics.mean(times)*1e3:>10.3f}"
              f"{max(times)*1e3:>10.3f}{dt/med:>12.1f}{len(times):>6}")
    return out


def main() -> None:
    print("=" * 74)
    print(" benchmark_engine.py -- cost of advance() by regime and chunk size")
    print(f" rsf_engine v{E.__version__}, formulation "
          f"{E.DEFAULT_FORMULATION}")
    print("=" * 74)
    print(" 'model/real' is model seconds simulated per second of wall time.")
    print(" A value of 1.0 means the simulation runs at real-time speed.")

    results = {}

    # ---- (1) STABLE: k = 3 kc_QD, creeping at V_lp ------------------------
    p = _params(3.0)
    eng = SpringSliderEngine(p, ConstantLoading(V_LP))
    st = eng.initial_state(V_fraction=1.0)
    st, _, _ = eng.advance(st, 200.0)          # settle onto steady creep
    results["stable"] = bench_regime(
        "(1) STABLE  k = 3.00 kc_QD, V ~ V_lp = 1e-6 m/s", eng, st, 30)

    # ---- (2) INTERSEISMIC of an unstable case -----------------------------
    p2 = _params(0.40)
    eng2 = SpringSliderEngine(p2, ConstantLoading(V_LP))
    st2 = eng2.initial_state(V_fraction=0.9)
    # Get past the spin-up event, then sit in the quiet phase.
    st2 = _advance_to_condition(eng2, st2, lambda v: v > 1e-3, dt=5.0)
    st2 = _advance_to_condition(eng2, st2, lambda v: v < 1e-8, dt=5.0)
    v_here = float(np.exp(st2.lnV))
    results["interseismic"] = bench_regime(
        f"(2) INTERSEISMIC  k = 0.40 kc_QD, V = {v_here:.3e} m/s "
        f"(locked limb)", eng2, st2, 30)

    # ---- (3) RAPID SLIP ---------------------------------------------------
    p3 = _params(0.40)
    eng3 = SpringSliderEngine(p3, ConstantLoading(V_LP))
    st3 = eng3.initial_state(V_fraction=0.9)
    # Approach the instability in stages. A 5 s chunk steps clean over a
    # whole ~0.03 s event, so the chunk size must shrink as V rises or the
    # state lands on the far side of the event instead of inside it.
    st3 = _advance_to_condition(eng3, st3, lambda v: v > 1e-3, dt=5.0)
    st3 = _advance_to_condition(eng3, st3, lambda v: v < 1e-8, dt=5.0)
    st3 = _advance_to_condition(eng3, st3, lambda v: v > 1e-7, dt=5.0)
    st3 = _advance_to_condition(eng3, st3, lambda v: v > 1e-5, dt=0.5)
    st3 = _advance_to_condition(eng3, st3, lambda v: v > 1e-2, dt=0.002)
    v3 = float(np.exp(st3.lnV))
    results["rapid"] = bench_regime(
        f"(3) RAPID SLIP  V = {v3:.3e} m/s (coseismic limb)", eng3, st3, 20)

    # ---- interpretation ---------------------------------------------------
    print("\n" + "=" * 74)
    print(" INTERPRETATION")
    print("=" * 74)
    worst = {}
    for dt in CHUNKS:
        worst[dt] = max(results[r][dt]["median"] for r in results)
    for dt in CHUNKS:
        budget_30 = 1.0 / 30.0
        print(f"  chunk {dt:>5.2f} s : worst-regime median "
              f"{worst[dt]*1e3:7.3f} ms  -> "
              f"{'fits' if worst[dt] < budget_30 else 'DOES NOT FIT'} in a "
              f"1/30 s frame budget ({budget_30*1e3:.1f} ms)")

    # Largest chunk that fits a 30 Hz budget in the worst regime
    ok = [dt for dt in CHUNKS if worst[dt] < 1.0 / 30.0]
    print(f"\n  Largest benchmarked chunk fitting a 30 Hz budget in the "
          f"worst regime: {max(ok) if ok else 'none'} s")
    print("  NOTE: the worst regime is the one that matters. A loop tuned on "
          "the\n  interseismic cost alone will stall the moment an event "
          "starts.")


if __name__ == "__main__":
    main()
