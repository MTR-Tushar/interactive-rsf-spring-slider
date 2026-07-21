"""
=============================================================================
 test_rsf_engine.py   --   v0.1   Stage-2 validation suite
=============================================================================
 Run:   python test_rsf_engine.py
 Deps:  numpy, scipy, rsf_engine.py in the same folder.

 WHAT THIS SUITE IS, AND WHAT IT IS NOT
 --------------------------------------
 It checks INTERNAL CONSISTENCY: that the code solves the equations it claims
 to solve, that analytically-derived quantities are reproduced numerically,
 that invariants hold, and that failures are reported rather than hidden.

 PASSING THESE TESTS IS NOT SCIENTIFIC VALIDATION. It does not show that the
 rate-and-state model, the ageing law, the radiation-damping approximation or
 the parameter values describe any real rock. Comparison against published
 results and against laboratory data remains future work.

 DECISION 2 OUTCOME (measured in section 12 below): formulation B,
 [ln V, ln theta, delta, u_lp], is selected. It gives a spring-stress
 residual 13x smaller (ageing law) to 64x smaller (slip law) than
 formulation A on an identical stick-slip sequence, and guarantees theta > 0
 through the coordinates rather than through an argument about the state
 equation, at a cost of roughly 6% more right-hand-side evaluations.
 Formulation A is retained and is exercised by every test here, so that any
 disagreement between the two is itself a diagnostic. The full reasoning,
 including the cancellation-error caveat that argues against B during rapid
 slip, is in the rsf_engine.py module docstring.

 A note on how stability is tested. The suite does NOT use "did events
 appear?" as evidence of instability. Threshold-crossing is an operational
 convention, and its absence over a finite window proves nothing. Instead
 every stability test perturbs the steady state by a relative 1e-6 and
 MEASURES the exponential growth or decay rate of the perturbation, then
 compares it with Re(lambda) from the analytically derived Jacobian.
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import rsf_engine as E
from rsf_engine import (ConstantLoading, EventDetector, History,
                        PiecewiseConstantLoading, RampLoading, RSFParams,
                        SimulationState, SinusoidalLoading, SpringSliderEngine,
                        StepLoading, stability_eigenvalues, stability_jacobian,
                        stability_summary)

# =============================================================================
# Minimal check harness (no pytest dependency: this must run on a bare Windows
# Python install with only numpy and scipy).
# =============================================================================

_RESULTS: List[Tuple[str, bool, str]] = []
_NOTES: List[str] = []
_WARNINGS: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(condition), detail))
    tag = "PASS" if condition else "FAIL"
    print(f"    [{tag}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def close(name: str, got: float, want: float, rtol: float = 1e-9,
          atol: float = 0.0) -> bool:
    ok = bool(np.isclose(got, want, rtol=rtol, atol=atol))
    rel = abs(got - want) / abs(want) if want != 0 else abs(got - want)
    return check(name, ok, f"got {got:.10g}, want {want:.10g}, rel {rel:.3e}")


def note(msg: str) -> None:
    _NOTES.append(msg)
    print(f"    NOTE: {msg}")


def warn(msg: str) -> None:
    _WARNINGS.append(msg)
    print(f"    WARNING: {msg}")


def section(title: str) -> None:
    print(f"\n{'-' * 74}\n{title}\n{'-' * 74}")


# =============================================================================
# Shared helpers
# =============================================================================

BASE = dict(mu0=0.60, Dc=20e-6, V0=1e-6, sigma_n=10e6, G=30e9, cs=3000.0)


def measure_growth_rate(params: RSFParams, V_lp: float, law: str,
                        formulation: str, pert: float = 1e-5,
                        rtol: float = 1e-11
                        ) -> Tuple[float, float, float]:
    """Perturb steady sliding and MEASURE the exponential rate of the
    perturbation, for comparison with Re(lambda) from the analytic Jacobian.

    Returns (measured_rate, predicted_rate, r2_of_fit), all in 1/s.

    WHY THE MEASUREMENT IS DONE IN A ROTATED BASIS. The loss of stability
    here is a Hopf bifurcation, so the eigenvalues are a complex pair
    lambda = mu +/- i*omega and the perturbation SPIRALS. In the raw
    coordinates (v, g) the quantity ln|(v,g)| is mu*t plus a bounded
    oscillation, so a straight-line fit only recovers mu if the window
    happens to span whole periods -- otherwise the fit is biased and its R^2
    is poor. That is a defect of the measurement, not of the solution.

    Instead, write the complex eigenvector as w = wr + i*wi. In the REAL
    basis {wr, wi} the linearised flow is exactly

        c(t) = exp(mu*t) * Rotation(omega*t) * c(0)

    so rho = |c| = sqrt(c1^2 + c2^2) is a PURE exponential with no
    oscillation at all, for any window length. Fitting ln rho therefore
    recovers mu directly and should give R^2 indistinguishable from 1. If
    the eigenvalues are real instead, the perturbation is launched along the
    dominant eigenvector and its coefficient is tracked on its own.

    The initial perturbation is launched along the relevant mode, is kept at
    1e-5 (so quadratic terms are ~1e-10 and the response is linear), and the
    fit is restricted to amplitudes between 1e-9 (above the solver noise
    floor) and 1e-3 (below the onset of nonlinearity).
    """
    J = stability_jacobian(params, V_lp)
    evals, evecs = np.linalg.eig(J)
    i = int(np.argmax(evals.real))
    lam = evals[i]
    mu_pred = float(lam.real)
    omega = float(abs(lam.imag))
    oscillatory = omega > 1e-12 * (abs(mu_pred) + 1.0)

    w = evecs[:, i]
    if oscillatory:
        B = np.column_stack([w.real, w.imag])
        direction = w.real
    else:
        j = 1 - i
        B = np.column_stack([evecs[:, i].real, evecs[:, j].real])
        direction = evecs[:, i].real
    direction = direction / np.linalg.norm(direction)
    Binv = np.linalg.inv(B)

    theta_ss = params.Dc / V_lp
    v0, g0 = pert * direction[0], pert * direction[1]

    # Window: aim for about 4 e-folds of change, which is ample for a fit
    # and keeps the amplitude inside the linear band.
    window = 4.0 / abs(mu_pred) if mu_pred != 0 else 8.0 * theta_ss
    window = max(window, 2.0 * theta_ss)

    sched = ConstantLoading(V_lp)
    eng = SpringSliderEngine(params, sched, law=law, formulation=formulation,
                             rtol=rtol, atol_lnV=1e-14, atol_state=1e-16,
                             min_samples_per_chunk=200)
    V_init = V_lp * (1.0 + v0)
    th_init = theta_ss * (1.0 + g0)
    st = SimulationState(t=0.0, lnV=float(np.log(V_init)), theta=th_init,
                         delta=0.0, u_lp=0.0, e0=0.0)
    tau0 = params.sigma_n * eng.friction(st.V, st.theta) + params.eta * st.V
    st.e0 = tau0 / params.k

    st, hist, reports = eng.run(st, window, window / 20.0)
    if not all(r.success for r in reports):
        return np.nan, mu_pred, 0.0

    a = hist.arrays()
    v = a["V"] / V_lp - 1.0
    g = a["theta"] / theta_ss - 1.0
    c = Binv @ np.vstack([v, g])
    rho = np.sqrt(c[0] ** 2 + c[1] ** 2) if oscillatory else np.abs(c[0])

    good = np.isfinite(rho) & (rho > 1e-9) & (rho < 1e-3)
    t, lr = a["t"][good], np.log(rho[good])
    if t.size < 30:
        return np.nan, mu_pred, 0.0
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, lr, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((lr - pred) ** 2))
    ss_tot = float(np.sum((lr - lr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(coef[0]), mu_pred, r2


def run_cycles(params: RSFParams, sched, law="ageing",
               formulation=E.DEFAULT_FORMULATION, t_total=3000.0,
               dt_chunk=5.0, threshold=1e-3, rtol=1e-9
               ) -> Tuple[History, EventDetector, List[E.StepReport]]:
    eng = SpringSliderEngine(params, sched, law=law, formulation=formulation,
                             rtol=rtol)
    st = eng.initial_state(V_fraction=0.9)
    hist = History()
    hist.append_initial(eng, st)
    det = EventDetector(threshold)
    reports = []
    n = int(np.ceil(t_total / dt_chunk))
    for _ in range(n):
        st, samples, rep = eng.advance(st, dt_chunk)
        hist.append(samples)
        det.feed(samples)
        reports.append(rep)
        if not rep.success:
            break
    det.finalise()
    return hist, det, reports


# =============================================================================
# TEST 1 -- analytical kc_QS
# =============================================================================

def test_kc_qs():
    section("TEST 1  --  Analytical quasi-static critical stiffness kc_QS")
    p = RSFParams(a=0.005, b=0.010, **BASE)
    # Worked example W1 of the project text: sigma_n = 10 MPa, b-a = 0.005,
    # Dc = 20 um  =>  kc = 2.5e9 Pa/m.
    close("kc_QS matches the worked value 2.5e9 Pa/m", p.kc_qs, 2.5e9,
          rtol=1e-12)
    # Dimensional/scaling behaviour
    p2 = RSFParams(a=0.005, b=0.010, **{**BASE, "sigma_n": 20e6})
    close("kc_QS doubles when sigma_n doubles", p2.kc_qs, 2.0 * p.kc_qs,
          rtol=1e-12)
    p3 = RSFParams(a=0.005, b=0.010, **{**BASE, "Dc": 40e-6})
    close("kc_QS halves when Dc doubles", p3.kc_qs, 0.5 * p.kc_qs, rtol=1e-12)
    p4 = RSFParams(a=0.010, b=0.005, **BASE)
    check("kc_QS < 0 for velocity strengthening (a > b)", p4.kc_qs < 0,
          f"kc_QS = {p4.kc_qs:.4e} Pa/m")


# =============================================================================
# TEST 2 -- independently derived kc_QD
# =============================================================================

def test_kc_qd_derivation():
    section("TEST 2  --  Quasi-dynamic critical stiffness kc_QD "
            "(independent derivation)")
    p = RSFParams(a=0.005, b=0.010, k=1.0e9, **BASE)
    V_ss = 2e-3

    # (a) The closed form must be the root of trace(J) = 0. Find that root
    #     NUMERICALLY, without using the closed form, by bisection on k.
    def trace_of(k_val: float) -> float:
        pk = E.replace(p, k=k_val)
        return float(np.trace(stability_jacobian(pk, V_ss)))

    lo, hi = 1e6, 1e11
    check("trace(J) changes sign across the bracket",
          trace_of(lo) * trace_of(hi) < 0,
          f"tr(lo) = {trace_of(lo):.4e}, tr(hi) = {trace_of(hi):.4e}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if trace_of(lo) * trace_of(mid) <= 0:
            hi = mid
        else:
            lo = mid
    k_root = 0.5 * (lo + hi)
    close("bisection root of trace(J)=0 equals the closed form kc_QD",
          k_root, p.kc_qd(V_ss), rtol=1e-10)

    # (b) determinant is positive everywhere -> the bifurcation is a Hopf,
    #     not a saddle-node. Sample widely.
    dets = []
    for k_val in np.logspace(6, 11, 60):
        for vss in np.logspace(-9, -2, 20):
            pk = E.replace(p, k=k_val)
            dets.append(float(np.linalg.det(stability_jacobian(pk, vss))))
    check("det(J) > 0 over 1200 (k, V_ss) samples -> Hopf, not saddle-node",
          min(dets) > 0, f"min det = {min(dets):.4e}")

    # (c) the ageing and slip laws must share the linearisation. The Jacobian
    #     is law-independent by construction, so verify against a numerical
    #     Jacobian of each law's full right-hand side instead.
    for law in ("ageing", "slip"):
        eng = SpringSliderEngine(p, ConstantLoading(V_ss), law=law,
                                 formulation="lnV_lntheta")
        theta_ss = p.Dc / V_ss
        y0 = np.array([np.log(V_ss), np.log(theta_ss), 0.0, 0.0])
        eng._vlp = ConstantLoading(V_ss).v_lp
        J_full = np.array(eng._jac(0.0, y0))
        # Convert d/dL, d/dM (log coords) to the (v, g) coords of the
        # analytical Jacobian: to first order dv = dL and dg = dM.
        J_red = J_full[:2, :2]
        J_ana = stability_jacobian(p, V_ss)
        err = float(np.max(np.abs(J_red - J_ana)) / np.max(np.abs(J_ana)))
        check(f"analytic 2x2 Jacobian matches the {law}-law code Jacobian",
              err < 1e-10, f"max relative difference {err:.3e}")

    # (d) kc_QD < kc_QS, i.e. radiation damping is stabilising
    check("kc_QD < kc_QS (radiation damping is stabilising)",
          p.kc_qd(V_ss) < p.kc_qs,
          f"kc_QD = {p.kc_qd(V_ss):.5e}, kc_QS = {p.kc_qs:.5e} Pa/m "
          f"({100*(1 - p.kc_qd(V_ss)/p.kc_qs):.2f}% lower at V_ss = "
          f"{V_ss:.1e} m/s)")
    note(f"At the laboratory rate V_ss = 1e-6 m/s the correction is "
         f"eta*V_ss = {p.eta*1e-6:.3g} Pa against sigma_n(b-a) = "
         f"{p.sigma_n*(p.b-p.a):.3g} Pa, i.e. kc_QD and kc_QS differ by "
         f"{100*(1 - p.kc_qd(1e-6)/p.kc_qs):.4f}%. The distinction is real "
         f"but numerically invisible at lab rates.")


# =============================================================================
# TEST 3 -- steady sliding is a fixed point
# =============================================================================

def test_steady_sliding():
    section("TEST 3  --  Steady sliding is an exact fixed point")
    p = RSFParams(a=0.005, b=0.010, k=1.0e9, **BASE)
    V_lp = 1e-6
    theta_ss = p.Dc / V_lp
    close("theta_ss = Dc/V_ss", theta_ss, 20.0, rtol=1e-12)

    for law in ("ageing", "slip"):
        for form in ("lnV_theta", "lnV_lntheta"):
            eng = SpringSliderEngine(p, ConstantLoading(V_lp), law=law,
                                     formulation=form)
            st = eng.initial_state(V_fraction=1.0)
            y = eng._to_vector(st)
            eng._vlp = eng.schedule.v_lp
            f = np.asarray(eng._rhs(0.0, y), dtype=float)
            scale = max(abs(f[2]), 1e-30)
            # Threshold is a few tens of machine epsilon. Going below eps
            # would test the floating-point unit, not the physics: the
            # steady state is only representable to within rounding.
            check(f"RHS of (lnV, state) vanishes at the fixed point "
                  f"[{law}, {form}]",
                  abs(f[0]) < 1e-14 and abs(f[1]) < 1e-14,
                  f"dlnV/dt = {f[0]:.3e} 1/s, dstate/dt = {f[1]:.3e} 1/s "
                  f"(machine eps = {np.finfo(float).eps:.2e})")
            check(f"ddelta/dt = V and du_lp/dt = V_lp [{law}, {form}]",
                  np.isclose(f[2], V_lp, rtol=1e-14)
                  and np.isclose(f[3], V_lp, rtol=1e-14),
                  f"ddelta/dt = {f[2]:.6e}, du_lp/dt = {f[3]:.6e} m/s")

    # V*theta/Dc must be exactly 1 at steady state
    close("V*theta/Dc = 1 at steady state", V_lp * theta_ss / p.Dc, 1.0,
          rtol=1e-14)


# =============================================================================
# TEST 4 -- velocity strengthening
# =============================================================================

def test_velocity_strengthening():
    section("TEST 4  --  Velocity strengthening (a > b)")
    p = RSFParams(a=0.010, b=0.005, k=1.0e9, **BASE)
    V_lp = 1e-6
    check("validate() reports velocity strengthening",
          any("VELOCITY STRENGTHENING" in w for w in p.validate(V_ss=V_lp)),
          "")
    m, pr, r2 = measure_growth_rate(p, V_lp, "ageing", "lnV_lntheta")
    check("predicted growth rate is negative (stable)", pr < 0,
          f"Re(lambda)max = {pr:.6e} 1/s")
    check("measured decay rate matches prediction within 1%",
          np.isfinite(m) and abs(m - pr) / abs(pr) < 0.01,
          f"measured {m:.6e}, predicted {pr:.6e} 1/s, fit R^2 = {r2:.6f}")
    # And a long nonlinear run must converge to V_lp
    hist, det, reps = run_cycles(p, ConstantLoading(V_lp), t_total=2000.0)
    a = hist.arrays()
    close("V converges to V_lp after 2000 s", float(a["V"][-1]), V_lp,
          rtol=1e-4)
    check("no events detected", len(det.events) == 0,
          f"{len(det.events)} events")


# =============================================================================
# TEST 5/6/7 -- stable, unstable and near-critical velocity weakening
# =============================================================================

def test_stability_regimes():
    section("TEST 5-7  --  Stable / unstable / near-critical "
            "velocity weakening (growth rates, not event counts)")
    V_lp = 1e-6
    cases = [
        ("clearly stable   k = 3.00 kc_QD", 3.00, False),
        ("near-critical    k = 1.02 kc_QD", 1.02, False),
        ("near-critical    k = 0.98 kc_QD", 0.98, True),
        ("clearly unstable k = 0.40 kc_QD", 0.40, True),
    ]
    for label, ratio, expect_unstable in cases:
        p = RSFParams.from_stiffness_ratio(ratio, "qd", V_ss=V_lp,
                                           a=0.005, b=0.010, **BASE)
        s = stability_summary(p, V_lp)
        pred_unstable = s["growth_rate"] > 0
        check(f"{label}: sign of Re(lambda) matches k vs kc_QD",
              pred_unstable == expect_unstable,
              f"Re(lambda)max = {s['growth_rate']:+.5e} 1/s, "
              f"k/kc_QD = {s['k_over_kc_qd']:.4f}")
        m, pr, r2 = measure_growth_rate(p, V_lp, "ageing", "lnV_lntheta")
        rel = abs(m - pr) / abs(pr) if pr != 0 else np.nan
        check(f"{label}: MEASURED rate matches predicted within 1%",
              np.isfinite(m) and rel < 0.01,
              f"measured {m:+.5e}, predicted {pr:+.5e} 1/s, "
              f"rel {rel:.3e}, R^2 {r2:.6f}")

    # Nonlinear consequence: the unstable case must reach a limit cycle.
    p_un = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                          a=0.005, b=0.010, **BASE)
    hist, det, reps = run_cycles(p_un, ConstantLoading(V_lp), t_total=4000.0)
    check("unstable case produces threshold-crossing events",
          len(det.events) >= 3, f"{len(det.events)} events at V > 1 mm/s")
    rec = det.recurrence_intervals(exclude_spin_up=True)
    if rec.size >= 2:
        spread = float(np.std(rec) / np.mean(rec))
        check("post-spin-up recurrence intervals are repeatable (CoV < 1%)",
              spread < 0.01,
              f"mean {np.mean(rec):.4f} s, CoV {spread:.3e}, n = {rec.size}")
        note(f"Limit-cycle recurrence time {np.mean(rec):.2f} s; peak V "
             f"{max(e.V_max for e in det.events):.4e} m/s.")

    # The stable velocity-weakening case must settle onto V_lp.
    p_st = RSFParams.from_stiffness_ratio(3.0, "qd", V_ss=V_lp,
                                          a=0.005, b=0.010, **BASE)
    hist2, det2, _ = run_cycles(p_st, ConstantLoading(V_lp), t_total=2000.0)
    a2 = hist2.arrays()
    close("stable velocity-weakening case creeps at V_lp",
          float(a2["V"][-1]), V_lp, rtol=1e-4)
    check("stable velocity-weakening case produces no events",
          len(det2.events) == 0, f"{len(det2.events)} events")


# =============================================================================
# TEST 8 -- the discriminating test: kc_QD vs kc_QS disagree
# =============================================================================

def test_kc_qd_discriminates():
    section("TEST 8  --  Discriminating test: k between kc_QD and kc_QS")
    V_lp = 2e-3        # high loading rate, so eta*V_ss is NOT negligible
    p = RSFParams.from_stiffness_ratio(1.10, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    r_qs = p.stiffness_ratio_qs()
    r_qd = p.stiffness_ratio_qd(V_lp)
    check("chosen k lies between kc_QD and kc_QS",
          r_qd > 1.0 > r_qs,
          f"k/kc_QS = {r_qs:.4f} (< 1: QS predicts UNSTABLE), "
          f"k/kc_QD = {r_qd:.4f} (> 1: QD predicts STABLE)")
    check("validate() flags the disagreement",
          any("CRITERIA DISAGREE" in w for w in p.validate(V_ss=V_lp)), "")

    m, pr, r2 = measure_growth_rate(p, V_lp, "ageing", "lnV_lntheta")
    check("the IMPLEMENTED system is stable -> kc_QD is the right threshold",
          np.isfinite(m) and m < 0 and pr < 0,
          f"measured {m:+.5e} 1/s, predicted {pr:+.5e} 1/s, R^2 {r2:.6f}")
    check("measured rate matches the kc_QD-based prediction within 1%",
          np.isfinite(m) and abs(m - pr) / abs(pr) < 0.01,
          f"rel difference {abs(m - pr)/abs(pr):.3e}")

    # Mirror case just BELOW kc_QD must be unstable.
    p2 = RSFParams.from_stiffness_ratio(0.90, "qd", V_ss=V_lp,
                                        a=0.005, b=0.010, **BASE)
    m2, pr2, _ = measure_growth_rate(p2, V_lp, "ageing", "lnV_lntheta")
    check("k = 0.90 kc_QD at the same V_lp is unstable",
          np.isfinite(m2) and m2 > 0 and pr2 > 0,
          f"measured {m2:+.5e} 1/s, predicted {pr2:+.5e} 1/s")
    note("This is the test that distinguishes the two thresholds. The "
         "quasi-static criterion kc_QS predicts instability for the first "
         "case and is contradicted by the implemented system.")


# =============================================================================
# TEST 9 -- ageing vs slip: same linear threshold, different nonlinear cycle
# =============================================================================

def test_ageing_vs_slip():
    section("TEST 9  --  Ageing vs slip law")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    m_ag, pr_ag, _ = measure_growth_rate(p, V_lp, "ageing", "lnV_lntheta")
    m_sl, pr_sl, _ = measure_growth_rate(p, V_lp, "slip", "lnV_lntheta")
    close("both laws share the PREDICTED linear growth rate", pr_sl, pr_ag,
          rtol=1e-14)
    check("both laws share the MEASURED linear growth rate within 2%",
          abs(m_sl - m_ag) / abs(m_ag) < 0.02,
          f"ageing {m_ag:+.5e}, slip {m_sl:+.5e} 1/s")

    h_ag, d_ag, _ = run_cycles(p, ConstantLoading(V_lp), law="ageing",
                               t_total=4000.0)
    h_sl, d_sl, _ = run_cycles(p, ConstantLoading(V_lp), law="slip",
                               t_total=4000.0)
    r_ag = d_ag.recurrence_intervals()
    r_sl = d_sl.recurrence_intervals()
    if r_ag.size and r_sl.size:
        diff = abs(np.mean(r_sl) - np.mean(r_ag)) / np.mean(r_ag)
        check("the two laws give DIFFERENT nonlinear limit cycles",
              diff > 1e-3,
              f"recurrence: ageing {np.mean(r_ag):.3f} s, "
              f"slip {np.mean(r_sl):.3f} s ({100*diff:.2f}% apart)")
    vmax_ag = max((e.V_max for e in d_ag.events), default=np.nan)
    vmax_sl = max((e.V_max for e in d_sl.events), default=np.nan)
    note(f"Peak slip velocity: ageing {vmax_ag:.4e} m/s, "
         f"slip {vmax_sl:.4e} m/s. Same linear threshold, different "
         f"nonlinear behaviour -- exactly as the project text states.")


# =============================================================================
# TEST 10 -- spring-stress invariant
# =============================================================================

def test_spring_invariant():
    section("TEST 10  --  Spring-stress invariant "
            "tau_spring - tau_total = 0")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    for form in ("lnV_theta", "lnV_lntheta"):
        hist, det, reps = run_cycles(p, ConstantLoading(V_lp),
                                     formulation=form, t_total=3000.0)
        a = hist.arrays()
        res = np.abs(a["residual"])
        rel = res.max() / p.sigma_n
        check(f"[{form}] max |tau_spring - tau_total| is at solver tolerance",
              rel < 1e-8,
              f"max {res.max():.4e} Pa = {rel:.3e} of sigma_n "
              f"(over {len(hist)} samples spanning "
              f"{a['t'][-1]-a['t'][0]:.0f} s)")
        # The invariant must also hold DURING events, not only interseismically
        ev_mask = a["V"] > 1e-3
        if ev_mask.any():
            rel_ev = float(res[ev_mask].max()) / p.sigma_n
            check(f"[{form}] invariant also holds during rapid slip",
                  rel_ev < 1e-8,
                  f"max {res[ev_mask].max():.4e} Pa = {rel_ev:.3e} of sigma_n")


# =============================================================================
# TEST 11 -- state positivity without clipping
# =============================================================================

def test_positivity():
    section("TEST 11  --  V > 0 and theta > 0, by coordinates not clipping")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    for form in ("lnV_theta", "lnV_lntheta"):
        for law in ("ageing", "slip"):
            hist, det, reps = run_cycles(p, ConstantLoading(V_lp), law=law,
                                         formulation=form, t_total=3000.0)
            a = hist.arrays()
            check(f"[{form}, {law}] V > 0 everywhere",
                  bool((a["V"] > 0).all()),
                  f"min V = {a['V'].min():.4e} m/s, "
                  f"max V = {a['V'].max():.4e} m/s")
            check(f"[{form}, {law}] theta > 0 everywhere",
                  bool((a["theta"] > 0).all()),
                  f"min theta = {a['theta'].min():.4e} s")
            check(f"[{form}, {law}] all outputs finite",
                  bool(np.isfinite(a["tau_total"]).all()
                       and np.isfinite(a["mu"]).all()), "")


# =============================================================================
# TEST 12 -- formulation comparison (Decision 2)
# =============================================================================

def test_formulation_comparison():
    section("TEST 12  --  Formulation A [lnV, theta] vs B [lnV, ln theta]")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    table = {}
    for form in ("lnV_theta", "lnV_lntheta"):
        for law in ("ageing", "slip"):
            t0 = time.perf_counter()
            hist, det, reps = run_cycles(p, ConstantLoading(V_lp), law=law,
                                         formulation=form, t_total=3000.0)
            wall = time.perf_counter() - t0
            a = hist.arrays()
            table[(form, law)] = dict(
                wall=wall, nfev=sum(r.nfev for r in reps),
                njev=sum(r.njev for r in reps), nlu=sum(r.nlu for r in reps),
                samples=len(hist),
                res=float(np.abs(a["residual"]).max() / p.sigma_n),
                theta_min=float(a["theta"].min()),
                V_max=float(a["V"].max()),
                n_ev=len(det.events),
                rec=float(np.mean(det.recurrence_intervals()))
                if det.recurrence_intervals().size else np.nan,
                ok=all(r.success for r in reps))
    print()
    print(f"    {'formulation':<14}{'law':<8}{'nfev':>8}{'njev':>7}"
          f"{'nlu':>7}{'wall/s':>9}{'rel resid':>12}{'theta_min/s':>13}"
          f"{'events':>8}")
    for (form, law), d in table.items():
        print(f"    {form:<14}{law:<8}{d['nfev']:>8}{d['njev']:>7}"
              f"{d['nlu']:>7}{d['wall']:>9.2f}{d['res']:>12.2e}"
              f"{d['theta_min']:>13.3e}{d['n_ev']:>8}")
    print()
    for key, d in table.items():
        check(f"{key[0]}/{key[1]} completed without solver failure", d["ok"],
              "")
    # Physical agreement between formulations
    for law in ("ageing", "slip"):
        rA = table[("lnV_theta", law)]["rec"]
        rB = table[("lnV_lntheta", law)]["rec"]
        if np.isfinite(rA) and np.isfinite(rB):
            rel = abs(rA - rB) / rA
            check(f"[{law}] both formulations give the same recurrence time",
                  rel < 1e-3,
                  f"A {rA:.5f} s vs B {rB:.5f} s, rel {rel:.3e}")
    return table


# =============================================================================
# TEST 13 -- loading-step breakpoint handling
# =============================================================================

def test_breakpoints():
    section("TEST 13  --  Schedule breakpoints and chunk splitting")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(3.0, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)

    # A step placed deliberately INSIDE a chunk.
    sched = StepLoading(V1=1e-6, V2=1e-5, t_step=7.3)
    eng = SpringSliderEngine(p, sched)
    st = eng.initial_state(V_fraction=1.0)
    st, samples, rep = eng.advance(st, 20.0)   # chunk 0 -> 20 contains 7.3
    check("chunk containing a step is split into 2 sub-intervals",
          rep.n_subintervals == 2, f"{rep.n_subintervals} sub-intervals")
    check("solver reported success across the step", rep.success, rep.message)
    t, vlp = samples["t"], samples["V_lp"]
    before = vlp[t < 7.3]
    after = vlp[t > 7.3]
    check("V_lp is exactly V1 at every sample before the step",
          bool(np.all(before == 1e-6)),
          f"{before.size} samples, unique {np.unique(before)}")
    check("V_lp is exactly V2 at every sample after the step",
          bool(np.all(after == 1e-5)),
          f"{after.size} samples, unique {np.unique(after)}")
    check("a sample sits exactly at the breakpoint",
          bool(np.any(np.isclose(t, 7.3, atol=1e-12))), "")

    # The direct effect must appear: mu jumps by a*ln(V2/V1) if V follows
    # V_lp quasi-statically. With a stiff spring (k = 3 kc_QD) the block
    # tracks the load point closely, so this is a meaningful comparison.
    mu = samples["mu"]
    i0 = int(np.searchsorted(t, 7.3) - 1)
    jump = float(mu[i0 + 1:].max() - mu[i0])
    predicted = p.a * np.log(1e-5 / 1e-6)
    check("friction rises after a step up in V_lp (direct effect present)",
          jump > 0, f"observed rise {jump:+.6f}, "
                    f"a*ln(V2/V1) = {predicted:+.6f} (upper bound: the "
                    f"spring is stiff but not rigid, so V lags V_lp)")
    check("observed rise does not exceed the rigid-machine direct effect",
          jump <= predicted * 1.05, "")

    # Ramp: two slope breakpoints.
    eng2 = SpringSliderEngine(p, RampLoading(1e-6, 5e-6, 2.0, 9.0))
    st2 = eng2.initial_state(V_fraction=1.0)
    st2, s2, rep2 = eng2.advance(st2, 15.0)
    check("ramp chunk is split at both slope breakpoints",
          rep2.n_subintervals == 3, f"{rep2.n_subintervals} sub-intervals")

    # Piecewise custom schedule.
    pw = PiecewiseConstantLoading(times=[0.0, 3.0, 6.0], values=[1e-6, 3e-6, 1e-6])
    eng3 = SpringSliderEngine(p, pw)
    st3 = eng3.initial_state(V_fraction=1.0)
    st3, s3, rep3 = eng3.advance(st3, 10.0)
    check("piecewise chunk is split at both switch times",
          rep3.n_subintervals == 3, f"{rep3.n_subintervals} sub-intervals")
    check("piecewise V_lp takes only the declared values",
          set(np.unique(s3["V_lp"])) <= {1e-6, 3e-6}, 
          f"unique values {np.unique(s3['V_lp'])}")

    # Sinusoid: no breakpoints, but a schedule-imposed max step.
    sn = SinusoidalLoading(V_mean=1e-6, V_amp=5e-7, period=50.0)
    check("sinusoid imposes its own max_step", sn.suggested_max_step() == 1.0,
          f"{sn.suggested_max_step()} s")
    try:
        SinusoidalLoading(V_mean=1e-6, V_amp=2e-6, period=50.0)
        check("sinusoid rejects amplitude >= mean (would need V_lp < 0)",
              False, "no exception raised")
    except ValueError:
        check("sinusoid rejects amplitude >= mean (would need V_lp < 0)",
              True, "")

    # Negative V_lp is prohibited outright.
    try:
        ConstantLoading(-1e-6).v_lp(0.0)
        check("negative V_lp is rejected", False, "no exception raised")
    except ValueError:
        check("negative V_lp is rejected", True, "")


# =============================================================================
# TEST 14 -- chunked vs single-shot
# =============================================================================

def test_chunked_vs_single_shot():
    section("TEST 14  --  Chunked integration vs single-shot")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    sched = ConstantLoading(V_lp)

    def final_state(dt_chunk, t_total, rtol=1e-11):
        eng = SpringSliderEngine(p, sched, rtol=rtol, atol_lnV=1e-13,
                                 atol_state=1e-15)
        st = eng.initial_state(V_fraction=0.9)
        n = int(round(t_total / dt_chunk))
        for _ in range(n):
            st, _, rep = eng.advance(st, dt_chunk)
            assert rep.success, rep.message
        return st

    # (a) INTERSEISMIC window, before any instability. Here the flow is
    #     contracting-ish and the comparison is meaningful.
    T = 100.0
    s_single = final_state(T, T)
    s_many = final_state(1.0, T)
    d_lnV = abs(s_single.lnV - s_many.lnV)
    d_delta = abs(s_single.delta - s_many.delta) / max(s_single.delta, 1e-30)
    check("interseismic: 100 chunks agree with 1 chunk in ln V",
          d_lnV < 1e-8, f"|d lnV| = {d_lnV:.3e}")
    check("interseismic: 100 chunks agree with 1 chunk in cumulative slip",
          d_delta < 1e-8, f"relative difference {d_delta:.3e}")

    # (b) ACROSS an instability. Report honestly rather than asserting.
    T2 = 1200.0
    a1 = final_state(T2, T2)
    a2 = final_state(2.0, T2)
    d2 = abs(a1.lnV - a2.lnV)
    note(f"Across a full stick-slip cycle (t = 0..{T2:.0f} s) the same "
         f"comparison gives |d lnV| = {d2:.3e}. Chunking does not change "
         f"the equations, but the unstable phase amplifies any difference "
         f"exponentially, so bitwise agreement across an event is not "
         f"expected and would not be a meaningful requirement.")
    check("across an event the trajectories remain physically equivalent "
          "(same order of magnitude of V)",
          abs(a1.lnV - a2.lnV) < 5.0,
          f"V single-shot {a1.V:.4e}, V chunked {a2.V:.4e} m/s")


# =============================================================================
# TEST 15 -- event detection, chunk-straddling and spin-up
# =============================================================================

def test_events():
    section("TEST 15  --  Incremental event detection and spin-up")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)

    # Same run, two very different chunk sizes: the catalogue must agree.
    cats = {}
    for dt in (5.0, 0.37):
        hist, det, reps = run_cycles(p, ConstantLoading(V_lp), t_total=3000.0,
                                     dt_chunk=dt)
        cats[dt] = det
    n5, n037 = len(cats[5.0].events), len(cats[0.37].events)
    check("event count is independent of chunk size", n5 == n037,
          f"dt=5.0 s -> {n5} events; dt=0.37 s -> {n037} events")
    if n5 == n037 and n5 >= 2:
        t5 = np.array([e.t_start for e in cats[5.0].events])
        t37 = np.array([e.t_start for e in cats[0.37].events])
        check("event onset times agree between chunk sizes (< 1 ms)",
              float(np.max(np.abs(t5 - t37))) < 1e-3,
              f"max difference {np.max(np.abs(t5 - t37)):.3e} s")

    det = cats[5.0]
    check("the first event is flagged as possible spin-up",
          det.events[0].is_possible_spin_up
          and not any(e.is_possible_spin_up for e in det.events[1:]),
          "")
    e0, e1 = det.events[0], det.events[1]
    rel = abs(e0.drop_tau_total_Pa - e1.drop_tau_total_Pa) \
        / e1.drop_tau_total_Pa
    check("spin-up event genuinely differs from the limit cycle",
          rel > 0.01,
          f"first drop_tau_total {e0.drop_tau_total_Pa/1e6:.4f} MPa vs "
          f"limit-cycle {e1.drop_tau_total_Pa/1e6:.4f} MPa "
          f"({100*rel:.1f}% apart)")
    r_in = det.recurrence_intervals(exclude_spin_up=False)
    r_ex = det.recurrence_intervals(exclude_spin_up=True)
    check("excluding the spin-up event changes recurrence statistics",
          r_in.size != r_ex.size,
          f"including: n = {r_in.size}, CoV "
          f"{np.std(r_in)/np.mean(r_in):.3e}; excluding: n = {r_ex.size}, "
          f"CoV {np.std(r_ex)/np.mean(r_ex):.3e}")

    # The three stress drops are reported separately and are NOT equal.
    e = det.events[-1] if not det.events[-1].truncated else det.events[-2]
    check("total, frictional and damping drops are reported separately "
          "and differ",
          abs(e.drop_tau_total_Pa - e.drop_tau_friction_Pa) > 1.0,
          f"total {e.drop_tau_total_Pa/1e6:.4f} MPa, "
          f"friction {e.drop_tau_friction_Pa/1e6:.4f} MPa, "
          f"damping {e.drop_tau_damping_Pa/1e6:.4f} MPa")
    note("Reporting a single unqualified 'stress drop' would be ambiguous: "
         "the radiation-damping component eta*V is a substantial fraction of "
         "the total during rapid slip.")

    # Threshold is a convention: changing it changes the catalogue.
    counts = {}
    for thr in (1e-4, 1e-3, 1e-2):
        _, d, _ = run_cycles(p, ConstantLoading(V_lp), t_total=3000.0,
                             threshold=thr)
        counts[thr] = (len(d.events),
                       float(np.mean([e.duration_s for e in d.events]))
                       if d.events else np.nan)
    note(f"Threshold sensitivity (events, mean duration): "
         + ", ".join(f"{t:.0e} m/s -> {c[0]} events, {c[1]:.4f} s"
                     for t, c in counts.items())
         + ". The threshold is an OPERATIONAL CONVENTION, not physics.")


# =============================================================================
# TEST 16 -- solver-failure reporting
# =============================================================================

def test_failure_reporting():
    section("TEST 16  --  Failure reporting (no silent corruption)")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)

    # (a) A deliberately corrupt state: theta = 0 makes dtheta/dt / theta
    #     infinite. The engine must REPORT, not raise and not return garbage
    #     dressed as success.
    eng = SpringSliderEngine(p, ConstantLoading(V_lp),
                             formulation="lnV_theta")
    bad = SimulationState(t=0.0, lnV=np.log(1e-6), theta=0.0, delta=0.0,
                          u_lp=0.0, e0=1e-3)
    try:
        st, samples, rep = eng.advance(bad, 1.0)
        check("corrupt state (theta = 0) is reported as a failure",
              not rep.success, f"message: {rep.message[:110]}")
        check("failure does not raise out of advance()", True, "")
    except Exception as exc:                                  # noqa: BLE001
        check("failure does not raise out of advance()", False,
              f"{type(exc).__name__}: {exc}")

    # (b) A stiffness so extreme that the problem becomes numerically brutal.
    p_hard = E.replace(p, k=1e-3)      # essentially no spring
    eng2 = SpringSliderEngine(p_hard, ConstantLoading(V_lp), rtol=1e-13,
                              atol_lnV=1e-16, atol_state=1e-18)
    st2 = eng2.initial_state(V_fraction=0.5)
    st2b, s2, rep2 = eng2.advance(st2, 1e5)
    check("extreme-stiffness run returns a StepReport either way",
          isinstance(rep2, E.StepReport),
          f"success = {rep2.success}, "
          f"{'message: ' + rep2.message[:80] if rep2.message else 'no message'}")
    if not rep2.success:
        check("failed run returns the LAST GOOD state, not a fabricated one",
              rep2.t_end <= st2.t + 1e5 and np.isfinite(st2b.lnV),
              f"t_end = {rep2.t_end:.6g} s, lnV = {st2b.lnV:.6g}")

    # (c) The residual warning must fire when the residual is large. Force it
    #     by starting with an inconsistent e0.
    eng3 = SpringSliderEngine(p, ConstantLoading(V_lp))
    st3 = eng3.initial_state(V_fraction=1.0)
    st3.e0 = st3.e0 * 1.5          # deliberately wrong preload
    st3b, s3, rep3 = eng3.advance(st3, 10.0)
    check("an inconsistent spring preload is caught by the residual monitor",
          any("residual" in w for w in rep3.warnings),
          f"max residual {rep3.max_abs_residual_Pa:.4e} Pa "
          f"({rep3.max_rel_residual:.3e} of sigma_n)")


# =============================================================================
# TEST 17 -- history structure, export, timeline reconstruction
# =============================================================================

def test_history_and_export():
    section("TEST 17  --  History, export and timeline reconstruction")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(0.40, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    hist, det, reps = run_cycles(p, ConstantLoading(V_lp), t_total=2000.0)
    a = hist.arrays()
    check("history time is strictly non-decreasing",
          bool(np.all(np.diff(a["t"]) >= 0)), f"{len(hist)} samples")
    check("history exposes every field the interface needs",
          set(E.FIELDS) == set(a.keys()), f"{len(E.FIELDS)} fields")

    # Timeline scrubbing: reconstruct an earlier picture from STORED data.
    t_target = 0.5 * float(a["t"][-1])
    i = hist.index_at_time(t_target)
    snap = hist.snapshot(i)
    check("index_at_time finds the nearest stored sample",
          abs(snap["t"] - t_target) < 1.0,
          f"requested {t_target:.3f} s, got {snap['t']:.3f} s")
    check("a reconstructed snapshot is internally consistent "
          "(spring extension matches stress)",
          abs(snap["tau_spring"] - snap["tau_total"]) / p.sigma_n < 1e-8,
          f"residual {snap['tau_spring']-snap['tau_total']:.3e} Pa; "
          f"e = {snap['e']*1e6:.4f} um, delta = {snap['delta']*1e6:.4f} um, "
          f"u_lp = {snap['u_lp']*1e6:.4f} um")

    dec = hist.decimate(500)
    check("decimation for plotting preserves endpoints and field set",
          dec["t"][0] == a["t"][0] and dec["t"][-1] == a["t"][-1]
          and set(dec.keys()) == set(a.keys()),
          f"{len(a['t'])} -> {len(dec['t'])} samples")

    outdir = os.path.dirname(os.path.abspath(__file__))
    csv_h = os.path.join(outdir, "_test_history.csv")
    csv_e = os.path.join(outdir, "_test_events.csv")
    hist.to_csv(csv_h, header_note="written by test_rsf_engine.py")
    det.to_csv(csv_e)
    ok_h = os.path.exists(csv_h) and os.path.getsize(csv_h) > 1000
    ok_e = os.path.exists(csv_e) and os.path.getsize(csv_e) > 100
    check("history CSV written with units in the header", ok_h,
          f"{os.path.getsize(csv_h)/1e6:.2f} MB" if ok_h else "missing")
    check("event-catalogue CSV written with explicit column names", ok_e,
          f"{os.path.getsize(csv_e)} bytes" if ok_e else "missing")
    if ok_h:
        with open(csv_h, encoding="utf-8") as f:
            head = [next(f) for _ in range(6)]
        check("history CSV header states the model's limitations",
              any("cannot compute nucleation length" in ln for ln in head), "")
    for pth in (csv_h, csv_e):
        try:
            os.remove(pth)
        except OSError:
            pass


# =============================================================================
# TEST 18 -- Hold experiment
# =============================================================================

def test_hold():
    section("TEST 18  --  Hold experiment (V_lp = 0)")
    V_lp = 1e-6
    p = RSFParams.from_stiffness_ratio(3.0, "qd", V_ss=V_lp,
                                       a=0.005, b=0.010, **BASE)
    sched = PiecewiseConstantLoading(times=[0.0, 50.0], values=[1e-6, 0.0])
    eng = SpringSliderEngine(p, sched)
    st = eng.initial_state(V_fraction=1.0)
    hist = History()
    hist.append_initial(eng, st)
    for _ in range(60):
        st, s, rep = eng.advance(st, 5.0)
        hist.append(s)
        if not rep.success:
            break
    a = hist.arrays()
    post = a["t"] > 50.0
    check("Hold runs without solver failure", rep.success, rep.message)
    check("during Hold the load point stops",
          bool(np.all(a["V_lp"][post] == 0.0)), "")
    check("during Hold the block still creeps (V > 0) and decelerates",
          bool(np.all(a["V"][post] > 0)) and a["V"][post][-1] < a["V"][post][0],
          f"V: {a['V'][post][0]:.4e} -> {a['V'][post][-1]:.4e} m/s")
    check("during Hold the spring unloads (tau_total falls)",
          a["tau_total"][post][-1] < a["tau_total"][post][0],
          f"tau_total: {a['tau_total'][post][0]/1e6:.4f} -> "
          f"{a['tau_total'][post][-1]/1e6:.4f} MPa")
    check("during Hold theta grows (ageing-law healing)",
          a["theta"][post][-1] > a["theta"][post][0],
          f"theta: {a['theta'][post][0]:.3f} -> {a['theta'][post][-1]:.3f} s")
    check("Hold schedule reports no steady state, so kc_QD is undefined",
          sched.steady_V() is None and not sched.is_stationary(), "")
    mu_min = float(a["mu"].min())
    if mu_min < 0.1:
        warn(f"friction fell to mu = {mu_min:.4f} during Hold: the "
             "unregularised logarithmic law is being pushed toward its "
             "domain of invalidity.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 74)
    print(f" test_rsf_engine.py  --  Stage-2 validation suite for "
          f"rsf_engine v{E.__version__}")
    print(f" numpy {np.__version__}   python {sys.version.split()[0]}")
    print("=" * 74)
    print(" These tests check INTERNAL CONSISTENCY of the implementation.")
    print(" They are NOT scientific validation of the physical model.")

    tests: List[Tuple[str, Callable]] = [
        ("kc_QS", test_kc_qs),
        ("kc_QD derivation", test_kc_qd_derivation),
        ("steady sliding", test_steady_sliding),
        ("velocity strengthening", test_velocity_strengthening),
        ("stability regimes", test_stability_regimes),
        ("kc_QD discriminating test", test_kc_qd_discriminates),
        ("ageing vs slip", test_ageing_vs_slip),
        ("spring invariant", test_spring_invariant),
        ("positivity", test_positivity),
        ("formulation comparison", test_formulation_comparison),
        ("breakpoints", test_breakpoints),
        ("chunked vs single-shot", test_chunked_vs_single_shot),
        ("events and spin-up", test_events),
        ("failure reporting", test_failure_reporting),
        ("history and export", test_history_and_export),
        ("hold", test_hold),
    ]
    t0 = time.perf_counter()
    crashed: List[str] = []
    for name, fn in tests:
        try:
            fn()
        except Exception:                                     # noqa: BLE001
            crashed.append(name)
            print(f"\n    !! TEST GROUP '{name}' RAISED:")
            traceback.print_exc()

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print("\n" + "=" * 74)
    print(f" SUMMARY: {n_pass} passed, {n_fail} failed, "
          f"{len(crashed)} test group(s) crashed, "
          f"{time.perf_counter() - t0:.1f} s total")
    print("=" * 74)
    if n_fail:
        print("\n FAILURES:")
        for nm, ok, det in _RESULTS:
            if not ok:
                print(f"   - {nm}: {det}")
    if _WARNINGS:
        print("\n NUMERICAL WARNINGS:")
        for w in _WARNINGS:
            print(f"   - {w}")
    print("\n REMINDER: passing these tests shows the code solves the "
          "equations\n it claims to solve. It does NOT show that those "
          "equations describe\n any real rock. Comparison with published "
          "results and laboratory\n data remains future work.")
    return 1 if (n_fail or crashed) else 0


if __name__ == "__main__":
    sys.exit(main())
