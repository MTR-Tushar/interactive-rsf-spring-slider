"""
=============================================================================
 rsf_engine.py   --   v0.1   (Stage 2: numerical engine only)
=============================================================================
 Zero-dimensional QUASI-DYNAMIC rate-and-state spring-slider engine.

 Prepared for: Mohammad Tawhidur Rahman Tushar
 Project: interactive educational rate-and-state spring-slider simulator.

 This module contains NO interface code. It is the single source of truth for
 the physics and the numerics. Any front end (Streamlit, NiceGUI, React, ...)
 must obtain every displayed number from this module.

-----------------------------------------------------------------------------
 SCOPE AND LIMITATIONS  (must be repeated verbatim in the user interface)
-----------------------------------------------------------------------------
 This is a ZERO-DIMENSIONAL model: one block, one slip value, one velocity.
 It CANNOT compute:
     - nucleation length L_c
     - rupture-front velocity V_r
     - spatial stress heterogeneity
     - fault-edge or sample-boundary effects
     - acoustic-emission locations
 Its "nucleation" is a TEMPORAL acceleration, not a spatial process.

 Radiation damping is NOT full inertia and NOT elastic-wave propagation. It
 is a local, instantaneous, velocity-proportional drag with no travel time,
 no interference and no reflections.

 Successful execution of this module is NOT scientific validation.

-----------------------------------------------------------------------------
 UNITS
-----------------------------------------------------------------------------
 STRICT SI THROUGHOUT. metres, seconds, pascals, kilograms.
 (The older rsf_velocity_step.py used micrometres. That convention is NOT
 used here. Unit conversion belongs to the presentation layer only.)

-----------------------------------------------------------------------------
 GOVERNING EQUATIONS
-----------------------------------------------------------------------------
 (1) Constitutive law  [model-dependent; standard one-state RSF]

         mu(V, theta) = mu0 + a*ln(V/V0) + b*ln(V0*theta/Dc)

     mu     friction coefficient                       [-]
     mu0    reference friction at V = V0               [-]
     a      direct-effect ("rate") parameter           [-]
     b      evolution-effect ("state") parameter       [-]
     V      LOCAL BLOCK SLIP VELOCITY                  [m/s]
     V0     reference velocity                         [m/s]
     theta  state variable: the evolving condition of the interface [s].
            NOT universally "contact age". The contact-age reading belongs
            to the ageing law and is not equally literal for every
            state-evolution law.
     Dc     critical slip distance                     [m]
     Dimensional check: V/V0 = 1 and V0*theta/Dc = (m/s * s)/m = 1, so both
     logarithms take dimensionless arguments. OK.

 (2) State evolution  [model-dependent; two standard choices]

         ageing (Dieterich):  dtheta/dt = 1 - V*theta/Dc
         slip   (Ruina):      dtheta/dt = -(V*theta/Dc)*ln(V*theta/Dc)

     Both sides have units of s/s = 1 for the ageing law (dtheta/dt is
     dimensionless because theta is in seconds). OK.

 (3) Spring (elastic loading)  [exact within the 0-D model]

         dtau/dt = k*(V_lp - V)

     tau    total driving shear stress                 [Pa]
     k      actual (loading) stiffness                 [Pa/m]
     V_lp   LOAD-POINT VELOCITY, imposed by the machine [m/s]
     Dimensional check: Pa/m * m/s = Pa/s. OK.

 (4) Quasi-dynamic force balance  [approximation]

         tau = sigma_n*mu(V,theta) + eta*V ,        eta = G/(2*cs)

     sigma_n effective normal stress                   [Pa]
     eta     radiation-damping coefficient             [Pa s/m]
     G       shear modulus                             [Pa]
     cs      shear-wave speed                          [m/s]
     Dimensional check: [eta] = Pa/(m/s) = Pa s/m; eta*V = Pa. OK.
     Status: APPROXIMATION. The factor 2 is conventional and mode-dependent.

 Differentiating (4) in time and substituting (3) gives the velocity ODE we
 actually integrate:

         dV/dt = [ k*(V_lp - V) - sigma_n*b*(dtheta/dt)/theta ]
                 / ( sigma_n*a/V + eta )

 and, because we integrate L = ln(V), the numerically preferable form

         dL/dt = [ k*(V_lp - V) - sigma_n*b*(dtheta/dt)/theta ]
                 / ( sigma_n*a + eta*V )                        (*)

 Note that (*) has no 1/V in the denominator at all, which is one reason the
 log-velocity formulation is well behaved over ten orders of magnitude in V.

-----------------------------------------------------------------------------
 SPRING PRELOAD CONVENTION  (Decision 1, recorded)
-----------------------------------------------------------------------------
 delta   cumulative BLOCK SLIP, with delta(0) = 0            [m]
 u_lp    LOAD-POINT DISPLACEMENT since Reset, du_lp/dt = V_lp,
         with u_lp(0) = 0                                    [m]
 tau_0   initial total driving stress
             = sigma_n*mu(V_init, theta_init) + eta*V_init   [Pa]
 e_0     initial spring PRE-EXTENSION = tau_0 / k            [m]
 e(t)    total physical spring extension = e_0 + u_lp - delta [m]
 tau_spring = k*e(t)                                          [Pa]

 The model is exactly consistent iff

     RESIDUAL(t) = tau_spring(t) - tau_constitutive(t) = 0

 where tau_constitutive = sigma_n*mu + eta*V. Because u_lp and delta are
 integrated INDEPENDENTLY of (L, theta), this residual is a genuine,
 free-of-charge numerical diagnostic: it is zero analytically, so whatever it
 shows is integration error. It is evaluated continuously and reported.

-----------------------------------------------------------------------------
 CRITICAL STIFFNESS: TWO DIFFERENT THRESHOLDS
-----------------------------------------------------------------------------
 kc_QS = sigma_n*(b - a)/Dc
     The standard QUASI-STATIC one-degree-of-freedom critical stiffness. It
     is the reference value quoted in the literature and in the project text.
     Same for the ageing and slip laws (identical first-order linearisation).

 kc_QD = [sigma_n*(b - a) - eta*V_ss] / Dc
     The threshold for the QUASI-DYNAMIC system ACTUALLY IMPLEMENTED HERE,
     derived below. Depends on the steady-sliding velocity V_ss, so it only
     exists when a constant steady loading state exists (V_lp constant).

 DERIVATION (recorded here so the code and the physics cannot drift apart).
 Let V = V_ss*(1+v), theta = theta_ss*(1+g), with theta_ss = Dc/V_ss so that
 V_ss*theta_ss/Dc = 1 exactly. Define

     s = V_ss/Dc = 1/theta_ss   [1/s]
     A = sigma_n*a + eta*V_ss   [Pa]

 State equation, AGEING law: dtheta/dt = 1 - V*theta/Dc. To first order
 V*theta/Dc = 1 + v + g, and dtheta/dt = theta_ss*dg/dt, hence
     dg/dt = -s*(v + g).
 State equation, SLIP law: with x = V*theta/Dc = 1 + v + g we have
 ln(x) = v + g and x*ln(x) = v + g to first order, so dtheta/dt = -(v+g)
 and again
     dg/dt = -s*(v + g).
 The two laws therefore share the same linearisation. This is the reason the
 standard linear critical stiffness does not distinguish them.

 Velocity equation, linearised from (*):
     A*dv/dt = (-k*V_ss + sigma_n*b*s)*v + sigma_n*b*s*g

 Jacobian of the reduced (v, g) system:

     J = [[ (-k*V_ss + sigma_n*b*s)/A ,  sigma_n*b*s/A ],
          [ -s                        , -s             ]]

     det J = s*k*V_ss/A  > 0   ALWAYS (k, V_ss, A all positive)
     tr  J = (-k*V_ss + sigma_n*b*s)/A - s

 Because det J > 0 for every admissible parameter set, stability is decided
 by the sign of the trace alone, and the loss of stability is a HOPF
 bifurcation (a complex-conjugate pair crossing the imaginary axis), not a
 saddle-node. Setting tr J = 0:

     -k*V_ss + sigma_n*b*s = s*(sigma_n*a + eta*V_ss)
     k*V_ss = (V_ss/Dc)*[sigma_n*(b-a) - eta*V_ss]
     k      = [sigma_n*(b-a) - eta*V_ss]/Dc  =  kc_QD          QED

 Dimensional check: sigma_n*(b-a) is Pa; eta*V_ss is (Pa s/m)*(m/s) = Pa;
 their difference divided by Dc [m] gives Pa/m, a stiffness. OK.

 CONSEQUENCE: kc_QD < kc_QS whenever eta > 0 and V_ss > 0, i.e. RADIATION
 DAMPING IS STABILISING. It shrinks the unstable window. At laboratory
 loading rates (V_lp ~ 1e-6 m/s) the correction eta*V_ss ~ 5 Pa is utterly
 negligible against sigma_n*(b-a) ~ 5e4 Pa, so kc_QD and kc_QS agree to about
 one part in 1e4. The distinction becomes real only at high V_lp.

 Status: [Model-dependent] Exact for the linearised, single-state-variable,
 quasi-dynamic, zero-dimensional spring-slider implemented here. Verified
 numerically in test_rsf_engine.py against measured perturbation growth and
 decay rates -- not merely against the presence or absence of events.

 For time-dependent loading schedules (ramp, sinusoid, piecewise) there is no
 stationary steady-sliding state, so NO single stationary critical-stiffness
 threshold applies. The interface must say so rather than displaying a
 misleading number.

-----------------------------------------------------------------------------
 STATE REPRESENTATION  (Decision 2, recorded)
-----------------------------------------------------------------------------
 Two formulations are implemented and both are exercised by the test suite.
 No clipping is used in either. Positivity is a property of the coordinates,
 not of a guard.

 FORMULATION A  "lnV_theta":    y = [ L, theta, delta, u_lp ],  L = ln V
     dL/dt     = ( k*(V_lp - V) - sigma_n*b*(dtheta/dt)/theta )
                 / ( sigma_n*a + eta*V )
     dtheta/dt = 1 - V*theta/Dc                       (ageing)
               = -x*ln(x),  x = V*theta/Dc            (slip)
     ddelta/dt = V
     du_lp/dt  = V_lp(t)
     V > 0 is guaranteed. theta > 0 is NOT guaranteed by the coordinates,
     although it is guaranteed by the ageing law itself (dtheta/dt = 1 > 0
     whenever theta -> 0) and by the slip law (dtheta/dt -> 0 as theta -> 0).

 FORMULATION B  "lnV_lntheta":  y = [ L, M, delta, u_lp ],  M = ln theta
     Transformed state equations (derived by dividing dtheta/dt by theta,
     since dM/dt = (dtheta/dt)/theta):
         ageing:  dM/dt = exp(-M) - exp(L)/Dc
                        = 1/theta - V/Dc
         slip:    dM/dt = -(V/Dc)*ln(V*theta/Dc)
                        = -(exp(L)/Dc)*(L + M - ln Dc)
     Dimensional check: [dM/dt] = 1/s for both; exp(-M) = 1/theta [1/s] and
     V/Dc = (m/s)/m = 1/s. OK.
     Both V > 0 and theta > 0 are guaranteed by the coordinates.

 SELECTION: see test_rsf_engine.py section 12, which runs an identical
 unstable cycle in both formulations under both laws and compares solver
 effort, minimum state values, the spring-stress residual and wall time. The
 outcome is recorded immediately below and in the README.

-----------------------------------------------------------------------------
 DECISION 2 OUTCOME  (recorded from measurement, not from preference)
-----------------------------------------------------------------------------
 Both formulations were run through an identical 3000 s stick-slip sequence
 (k = 0.40 kc_QD, V_lp = 1e-6 m/s) under both state-evolution laws. Measured:

   formulation   law        nfev   njev    nlu  wall/s   rel resid  theta_min/s
   lnV_theta     ageing    80028    993  12388    2.59    9.54e-12    2.069e-04
   lnV_theta     slip      94474   1106  12770    3.71    1.87e-11    1.925e-04
   lnV_lntheta   ageing    85126    890  12318    2.89    7.16e-13    2.069e-04
   lnV_lntheta   slip      94625    982  12284    3.10    2.94e-13    1.925e-04

 SELECTED: FORMULATION B, "lnV_lntheta"  ->  y = [ln V, ln theta, delta, u_lp]

 Reasons, in order of weight:
 1. Accuracy. The spring-stress residual -- which is analytically zero, so it
    measures nothing but integration error -- is 13x smaller for the ageing
    law and 64x smaller for the slip law in formulation B. This is the only
    independent accuracy diagnostic available, and it favours B decisively.
 2. Positivity by construction. In B both V > 0 and theta > 0 are properties
    of the COORDINATES: theta = exp(M) cannot be non-positive whatever the
    solver does. In A, theta is a raw state component and its positivity
    relies on the state equation continuing to protect it. Neither
    formulation uses clipping, so B is the one that needs no such argument.
 3. The fixed point is represented more exactly. At steady sliding the
    ageing-law residual of A is -4.44e-16 (machine epsilon on a quantity of
    order 1, from evaluating 1 - V*theta/Dc), whereas B returns values at or
    below 6e-17. Small, but it is free.
 4. Fewer Jacobian evaluations and LU factorisations in both laws, i.e. B is
    marginally better conditioned for the implicit solver.

 The cost is 6% more right-hand-side evaluations for the ageing law and
 essentially none for the slip law -- wall-clock times are within the noise
 of one another. That is a cheap price for an order of magnitude in accuracy.

 CAVEAT, recorded honestly: the ageing law in B reads
 dM/dt = exp(-M) - exp(L)/Dc, and during rapid slip these two terms are both
 large and nearly cancel (theta_min = 2.07e-04 s gives exp(-M) ~ 4.8e3, while
 V/Dc ~ 5e3), so B is exposed to cancellation error exactly where A is not.
 The measurements above show this does not dominate at the tolerances used,
 but it is the reason both formulations are KEPT in the code rather than one
 being deleted. If a future parameter set makes B misbehave, A is available
 as a cross-check, and any disagreement between them is itself diagnostic.
 Both give the same limit-cycle recurrence time to 1 part in 1e8.
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

__version__ = "0.2"

# ---------------------------------------------------------------------------
# CHANGES IN v0.2 (Stage 3). Two additions only; no equation was touched and
# no default behaviour changed. All 107 v0.1 checks still pass unmodified.
#
# 1. SpringSliderEngine.set_schedule(). Stage 3 must change V_lp DURING a run
#    without resetting the state. Physically the swap must happen exactly at a
#    chunk boundary, which is automatic because advance() begins a fresh
#    solve_ivp at the current state every time it is called. The interface
#    could have written `engine.schedule = new`, but that is an undocumented
#    reach into the engine's internals and would silently leave the cached
#    _vlp binding stale. set_schedule() makes the operation explicit,
#    validates the new schedule, and rebinds _vlp. It does NOT touch lnV,
#    theta, delta, u_lp or e0 -- changing the loading is not allowed to move
#    the state.
#
# 2. SimulationState.copy(). Timeline handling and the controller's tests need
#    an independent copy of a state; constructing one by unpacking as_tuple()
#    in interface code invites argument-order mistakes.
# ---------------------------------------------------------------------------

# Formulation B. Selected on the Stage-2 measurements recorded in the module
# docstring under "DECISION 2 OUTCOME": 13x-64x smaller spring-stress residual
# than formulation A, positivity guaranteed by the coordinates, at a cost of
# about 6% more right-hand-side evaluations. Formulation A is retained and
# fully tested as an independent cross-check.
DEFAULT_FORMULATION = "lnV_lntheta"

# Operational convention, NOT a physical constant. Romanet et al. (2018) are
# often cited for a 1 mm/s separation between earthquakes and slow slip; that
# attribution is flagged as UNVERIFIED in the project text and the threshold
# is adjustable everywhere in this module.
DEFAULT_EVENT_THRESHOLD = 1e-3  # [m/s]


# =============================================================================
# 1. PARAMETERS
# =============================================================================

@dataclass
class RSFParams:
    """Rate-and-state spring-slider parameters. STRICT SI UNITS.

    Every field carries its physical meaning and unit. `k` is the actual
    loading stiffness in Pa/m; use `from_stiffness_ratio` if you would rather
    specify k as a multiple of a critical stiffness.
    """

    mu0: float = 0.60          # reference friction at V = V0            [-]
    a: float = 0.005           # direct-effect parameter                 [-]
    b: float = 0.010           # evolution-effect parameter              [-]
    Dc: float = 20e-6          # critical slip distance                  [m]
    V0: float = 1e-6           # reference velocity                      [m/s]
    sigma_n: float = 10e6      # EFFECTIVE normal stress                 [Pa]
    G: float = 30e9            # shear modulus                           [Pa]
    cs: float = 3000.0         # shear-wave speed                        [m/s]
    k: float = 1.0e9           # actual loading stiffness                [Pa/m]

    # ---- derived ----------------------------------------------------------
    @property
    def eta(self) -> float:
        """Radiation-damping coefficient eta = G/(2 cs)  [Pa s/m].
        APPROXIMATION; the factor 2 is conventional and mode-dependent."""
        return self.G / (2.0 * self.cs)

    @property
    def kc_qs(self) -> float:
        """Quasi-static critical stiffness sigma_n(b-a)/Dc  [Pa/m].
        Negative if a > b (velocity strengthening): then no positive loading
        stiffness can destabilise the surface within this model."""
        return self.sigma_n * (self.b - self.a) / self.Dc

    def kc_qd(self, V_ss: float) -> float:
        """Quasi-dynamic critical stiffness of the IMPLEMENTED system,
        [sigma_n(b-a) - eta*V_ss]/Dc  [Pa/m].

        Only meaningful when a constant steady-sliding state exists, i.e.
        V_lp constant, in which case V_ss = V_lp. Derived in the module
        docstring; verified numerically in the test suite.
        """
        return (self.sigma_n * (self.b - self.a) - self.eta * V_ss) / self.Dc

    @property
    def a_minus_b(self) -> float:
        return self.a - self.b

    def stiffness_ratio_qs(self) -> float:
        return self.k / self.kc_qs if self.kc_qs != 0.0 else np.inf

    def stiffness_ratio_qd(self, V_ss: float) -> float:
        kqd = self.kc_qd(V_ss)
        return self.k / kqd if kqd != 0.0 else np.inf

    @classmethod
    def from_stiffness_ratio(cls, ratio: float, reference: str = "qs",
                             V_ss: Optional[float] = None, **kwargs) -> "RSFParams":
        """Build parameters with k set to `ratio` times kc_QS or kc_QD.

        reference: "qs" -> k = ratio * kc_QS
                   "qd" -> k = ratio * kc_QD(V_ss)   (V_ss required)
        """
        p = cls(**kwargs)
        if reference == "qs":
            kref = p.kc_qs
        elif reference == "qd":
            if V_ss is None:
                raise ValueError("reference='qd' requires V_ss (the steady "
                                 "sliding velocity, normally V_lp).")
            kref = p.kc_qd(V_ss)
        else:
            raise ValueError("reference must be 'qs' or 'qd'")
        return replace(p, k=ratio * kref)

    # ---- validation -------------------------------------------------------
    def validate(self, V_ss: Optional[float] = None) -> List[str]:
        """Return a list of human-readable warnings. Never raises for
        physically unusual but legal parameter sets: it explains them."""
        w: List[str] = []
        if self.a <= 0:
            w.append("a <= 0: the direct effect is non-positive. The "
                     "denominator sigma_n*a + eta*V may vanish; this is "
                     "outside standard RSF.")
        if self.b <= 0:
            w.append("b <= 0: no state-evolution weakening is possible.")
        if self.Dc <= 0:
            w.append("Dc <= 0 is not physical; the state equation is undefined.")
        if self.sigma_n <= 0:
            w.append("sigma_n <= 0 is not physical for a frictional contact.")
        if self.V0 <= 0:
            w.append("V0 <= 0 is not physical; ln(V/V0) is undefined.")
        if self.k <= 0:
            w.append("k <= 0: a non-positive loading stiffness is outside the "
                     "spring-slider model.")
        if self.G <= 0 or self.cs <= 0:
            w.append("G and cs must be positive for eta = G/(2 cs).")

        if self.a >= self.b:
            w.append(
                f"a - b = {self.a_minus_b:+.4f} >= 0: VELOCITY STRENGTHENING. "
                f"kc_QS = {self.kc_qs:.3e} Pa/m <= 0, so no positive loading "
                "stiffness satisfies k < kc within this model. Expect stable "
                "creep at V_lp. (This does not mean such a patch cannot be "
                "crossed by an externally driven rupture -- that question is "
                "outside a 0-D model.)")
        else:
            r_qs = self.stiffness_ratio_qs()
            if r_qs >= 1.0:
                w.append(
                    f"k/kc_QS = {r_qs:.4f} >= 1: the quasi-static criterion "
                    "predicts STABLE creep at V_lp despite velocity "
                    "weakening.")
            if V_ss is not None:
                kqd = self.kc_qd(V_ss)
                if kqd <= 0:
                    w.append(
                        f"kc_QD = {kqd:.3e} Pa/m <= 0 at V_ss = {V_ss:.3e} m/s: "
                        "at this loading rate radiation damping alone "
                        "stabilises the system for any positive k.")
                else:
                    r_qd = self.k / kqd
                    if (r_qs < 1.0) != (r_qd < 1.0):
                        w.append(
                            f"THE TWO CRITERIA DISAGREE at V_ss = {V_ss:.3e} m/s: "
                            f"k/kc_QS = {r_qs:.4f}, k/kc_QD = {r_qd:.4f}. The "
                            "implemented system follows kc_QD.")
        return w

    def summary(self, V_ss: Optional[float] = None) -> str:
        lines = [
            f"mu0     = {self.mu0:.4f}        [-]",
            f"a       = {self.a:.5f}       [-]",
            f"b       = {self.b:.5f}       [-]",
            f"a - b   = {self.a_minus_b:+.5f}      [-]  "
            f"({'velocity weakening' if self.a < self.b else 'velocity strengthening'})",
            f"Dc      = {self.Dc:.4e}   [m]",
            f"V0      = {self.V0:.4e}   [m/s]",
            f"sigma_n = {self.sigma_n:.4e}   [Pa]   (EFFECTIVE normal stress)",
            f"G       = {self.G:.4e}   [Pa]",
            f"cs      = {self.cs:.4e}   [m/s]",
            f"eta     = {self.eta:.4e}   [Pa s/m]  = G/(2 cs)",
            f"k       = {self.k:.4e}   [Pa/m]",
            f"kc_QS   = {self.kc_qs:.4e}   [Pa/m]  = sigma_n(b-a)/Dc",
            f"k/kc_QS = {self.stiffness_ratio_qs():.6f} [-]",
        ]
        if V_ss is not None:
            lines += [
                f"V_ss    = {V_ss:.4e}   [m/s]",
                f"kc_QD   = {self.kc_qd(V_ss):.4e}   [Pa/m]  "
                "= [sigma_n(b-a) - eta V_ss]/Dc",
                f"k/kc_QD = {self.stiffness_ratio_qd(V_ss):.6f} [-]",
                f"eta*V_ss= {self.eta * V_ss:.4e}   [Pa]  "
                f"(vs sigma_n(b-a) = {self.sigma_n*(self.b-self.a):.4e} Pa)",
            ]
        return "\n".join(lines)


# =============================================================================
# 2. LOADING SCHEDULES
# =============================================================================
#
# A schedule supplies V_lp(t) [m/s] and, crucially, the exact times at which
# V_lp or its derivative is discontinuous. The integrator must NEVER step
# across such a time: a stiff solver that straddles a jump will either fail or
# silently smear the jump over a step. All schedules enforce V_lp >= 0.
#
# NEGATIVE V_lp IS PROHIBITED in v0.1. The logarithmic constitutive law
# requires V > 0; reverse slip would need a signed-velocity friction
# formulation (e.g. a regularised arcsinh form with a sign function), which is
# a different model and is outside this version. V_lp = 0 is permitted only
# through an explicitly labelled Hold experiment.
# =============================================================================

class LoadingSchedule:
    """Base class. Subclasses implement v_lp(t) and breakpoints(t0, t1)."""

    name = "base"

    def v_lp(self, t: float) -> float:
        raise NotImplementedError

    def breakpoints(self, t0: float, t1: float) -> List[float]:
        """Times strictly inside (t0, t1) where V_lp or dV_lp/dt jumps."""
        return []

    def branch(self, ta: float, tb: float) -> Callable[[float], float]:
        """Return V_lp(t) with the correct BRANCH selected for the
        sub-interval (ta, tb), which by construction contains no breakpoint.

        This matters at the endpoints. A stiff collocation method evaluates
        the right-hand side AT tb; if tb is a jump time, the naive v_lp(tb)
        would return the value belonging to the NEXT sub-interval and
        contaminate the last step. Selecting the branch from the midpoint
        removes that ambiguity entirely. Continuous schedules (ramp,
        sinusoid) need no special treatment and inherit this default."""
        return self.v_lp

    def suggested_max_step(self) -> Optional[float]:
        """Upper bound on the solver step imposed by the schedule itself
        (e.g. to resolve a sinusoid). None means no schedule-driven limit."""
        return None

    def is_stationary(self) -> bool:
        """True if V_lp is constant for all time, so that a steady-sliding
        state -- and therefore kc_QD -- is defined."""
        return False

    def steady_V(self) -> Optional[float]:
        return None

    def describe(self) -> str:
        return self.name

    def _check(self, v: float) -> float:
        if v < 0.0:
            raise ValueError(
                f"{self.name}: V_lp = {v:.4e} m/s is negative. Reverse "
                "loading is not supported in v0.1: the logarithmic "
                "constitutive law requires V > 0, and reverse slip requires a "
                "signed-velocity friction formulation outside this model.")
        return float(v)


@dataclass
class ConstantLoading(LoadingSchedule):
    """V_lp(t) = V_const. Use V_const = 0.0 for a HOLD experiment."""
    V_const: float = 1e-6
    name: str = field(default="constant", init=False)

    def v_lp(self, t: float) -> float:
        return self._check(self.V_const)

    def is_stationary(self) -> bool:
        return self.V_const > 0.0

    def steady_V(self) -> Optional[float]:
        return self.V_const if self.V_const > 0.0 else None

    def describe(self) -> str:
        if self.V_const == 0.0:
            return ("HOLD: V_lp = 0 m/s. The load point is stationary; the "
                    "block may continue to creep, unloading the spring, and "
                    "under the ageing law the state keeps growing. No steady "
                    "sliding state exists, so kc_QD is undefined.")
        return f"constant V_lp = {self.V_const:.4e} m/s"


@dataclass
class StepLoading(LoadingSchedule):
    """V_lp = V1 for t < t_step, then V2. One breakpoint at t_step."""
    V1: float = 1e-6
    V2: float = 1e-5
    t_step: float = 100.0
    name: str = field(default="step", init=False)

    def v_lp(self, t: float) -> float:
        return self._check(self.V1 if t < self.t_step else self.V2)

    def breakpoints(self, t0: float, t1: float) -> List[float]:
        return [self.t_step] if t0 < self.t_step < t1 else []

    def branch(self, ta: float, tb: float) -> Callable[[float], float]:
        v = self.V1 if 0.5 * (ta + tb) < self.t_step else self.V2
        self._check(v)
        return lambda t, v=v: v

    def describe(self) -> str:
        return (f"step: V_lp = {self.V1:.3e} -> {self.V2:.3e} m/s "
                f"at t = {self.t_step:.4g} s")


@dataclass
class RampLoading(LoadingSchedule):
    """Linear ramp from V1 at t_start to V2 at t_end; constant outside.
    Breakpoints at t_start and t_end (slope discontinuities)."""
    V1: float = 1e-6
    V2: float = 1e-5
    t_start: float = 0.0
    t_end: float = 500.0
    name: str = field(default="ramp", init=False)

    def v_lp(self, t: float) -> float:
        if t <= self.t_start:
            v = self.V1
        elif t >= self.t_end:
            v = self.V2
        else:
            f = (t - self.t_start) / (self.t_end - self.t_start)
            v = self.V1 + f * (self.V2 - self.V1)
        return self._check(v)

    def breakpoints(self, t0: float, t1: float) -> List[float]:
        return [bp for bp in (self.t_start, self.t_end) if t0 < bp < t1]

    def describe(self) -> str:
        return (f"ramp: V_lp {self.V1:.3e} -> {self.V2:.3e} m/s over "
                f"t = {self.t_start:.4g}..{self.t_end:.4g} s")


@dataclass
class SinusoidalLoading(LoadingSchedule):
    """V_lp = V_mean + V_amp*sin(2 pi t / period).

    V_amp < V_mean is REQUIRED so that V_lp stays strictly positive: a
    sinusoid that dips to zero or below would demand reverse loading, which
    v0.1 prohibits."""
    V_mean: float = 1e-6
    V_amp: float = 5e-7
    period: float = 200.0
    name: str = field(default="sinusoid", init=False)

    def __post_init__(self):
        if self.V_amp >= self.V_mean:
            raise ValueError(
                f"SinusoidalLoading requires V_amp ({self.V_amp:.3e}) < "
                f"V_mean ({self.V_mean:.3e}) so that V_lp > 0 at all times. "
                "Reverse loading is outside v0.1.")
        if self.period <= 0:
            raise ValueError("period must be positive.")

    def v_lp(self, t: float) -> float:
        return self._check(
            self.V_mean + self.V_amp * np.sin(2.0 * np.pi * t / self.period))

    def suggested_max_step(self) -> Optional[float]:
        return self.period / 50.0   # resolve the sinusoid; smooth, no breaks

    def describe(self) -> str:
        return (f"sinusoid: V_lp = {self.V_mean:.3e} + {self.V_amp:.3e}"
                f"*sin(2 pi t/{self.period:.4g} s) m/s")


@dataclass
class PiecewiseConstantLoading(LoadingSchedule):
    """Custom piecewise-constant schedule.

    times:  non-decreasing switch times [s], times[0] should be <= 0
    values: V_lp on [times[i], times[i+1]) [m/s]
    """
    times: Sequence[float] = (0.0,)
    values: Sequence[float] = (1e-6,)
    name: str = field(default="piecewise", init=False)

    def __post_init__(self):
        self.times = list(map(float, self.times))
        self.values = list(map(float, self.values))
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have equal length.")
        if any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing.")
        for v in self.values:
            self._check(v)

    def v_lp(self, t: float) -> float:
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        i = max(0, min(i, len(self.values) - 1))
        return self._check(self.values[i])

    def breakpoints(self, t0: float, t1: float) -> List[float]:
        return [bp for bp in self.times if t0 < bp < t1]

    def branch(self, ta: float, tb: float) -> Callable[[float], float]:
        v = self.v_lp(0.5 * (ta + tb))
        return lambda t, v=v: v

    def describe(self) -> str:
        seg = ", ".join(f"t>={t:.4g}s: {v:.3e}"
                        for t, v in zip(self.times, self.values))
        return f"piecewise-constant V_lp [m/s]: {seg}"


# =============================================================================
# 3. SIMULATION STATE
# =============================================================================

@dataclass
class SimulationState:
    """Complete restart-capable snapshot. Everything the engine needs to
    continue is here; nothing else is hidden in the engine.

    t      simulation time since Reset                 [s]
    lnV    ln of local block slip velocity, V = exp(lnV) [-]
    theta  state variable                              [s]
    delta  cumulative block slip, delta(0) = 0         [m]
    u_lp   load-point displacement since Reset         [m]
    e0     initial spring pre-extension tau_0/k, FROZEN at Reset [m]
    """
    t: float
    lnV: float
    theta: float
    delta: float
    u_lp: float
    e0: float

    @property
    def V(self) -> float:
        return float(np.exp(self.lnV))

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.t, self.lnV, self.theta, self.delta, self.u_lp, self.e0)

    def copy(self) -> "SimulationState":
        """Independent copy. Used for timeline handling and for benchmarking
        the same starting point repeatedly."""
        return SimulationState(t=self.t, lnV=self.lnV, theta=self.theta,
                               delta=self.delta, u_lp=self.u_lp, e0=self.e0)


@dataclass
class StepReport:
    """Outcome of one call to advance(). ALWAYS inspect .success."""
    success: bool
    message: str
    t_start: float
    t_end: float
    n_subintervals: int
    n_samples: int
    nfev: int
    njev: int
    nlu: int
    wall_time_s: float
    max_abs_residual_Pa: float
    max_rel_residual: float          # residual / sigma_n
    min_V: float
    max_V: float
    min_theta: float
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = "OK " if self.success else "FAIL"
        return (f"[{head}] t {self.t_start:.6g} -> {self.t_end:.6g} s | "
                f"{self.n_samples} samples in {self.n_subintervals} "
                f"sub-interval(s) | nfev={self.nfev} njev={self.njev} | "
                f"|residual|max={self.max_abs_residual_Pa:.3e} Pa "
                f"({self.max_rel_residual:.2e} of sigma_n) | "
                f"V in [{self.min_V:.3e}, {self.max_V:.3e}] m/s | "
                f"theta_min={self.min_theta:.3e} s | "
                f"{self.wall_time_s*1e3:.1f} ms"
                + (f" | {self.message}" if self.message else ""))


# =============================================================================
# 4. THE ENGINE
# =============================================================================

class SpringSliderEngine:
    """Quasi-dynamic RSF spring-slider. Interface-free.

    Typical use:
        eng   = SpringSliderEngine(params, ConstantLoading(1e-6))
        state = eng.initial_state()
        while running:
            state, samples, report = eng.advance(state, dt_chunk=1.0)
            if not report.success:
                stop_and_report(report)
    """

    def __init__(self,
                 params: RSFParams,
                 schedule: LoadingSchedule,
                 law: str = "ageing",
                 formulation: str = DEFAULT_FORMULATION,
                 rtol: float = 1e-9,
                 atol_lnV: float = 1e-10,
                 atol_state: float = 1e-12,
                 atol_disp: float = 1e-15,
                 method: str = "Radau",
                 min_samples_per_chunk: int = 8):
        if law not in ("ageing", "slip"):
            raise ValueError("law must be 'ageing' or 'slip'")
        if formulation not in ("lnV_theta", "lnV_lntheta"):
            raise ValueError("formulation must be 'lnV_theta' or 'lnV_lntheta'")
        self.p = params
        self.schedule = schedule
        self.law = law
        self.formulation = formulation
        self.rtol = rtol
        self.atol_lnV = atol_lnV
        self.atol_state = atol_state
        self.atol_disp = atol_disp
        self.method = method
        self.min_samples_per_chunk = min_samples_per_chunk
        # Bound per sub-interval by advance(); the default is the raw
        # schedule, used by initial_state() and by direct rhs calls.
        self._vlp: Callable[[float], float] = schedule.v_lp

    # ---- initial condition ------------------------------------------------
    def initial_state(self,
                      V_init: Optional[float] = None,
                      theta_init: Optional[float] = None,
                      V_fraction: float = 1.0) -> SimulationState:
        """Build the state at t = 0 (Reset).

        By default the block starts at the steady state corresponding to the
        initial load-point velocity: V = V_lp(0), theta = Dc/V. That is an
        exact fixed point when V_lp is constant, so nothing will happen unless
        it is perturbed -- use V_fraction != 1 (e.g. 0.9) to start slightly
        off the fixed point, which is what makes an instability grow.

        delta(0) = 0 and u_lp(0) = 0 by convention (Decision 1). The initial
        spring pre-extension e0 = tau_0/k is computed from the initial state
        and then FROZEN: it is the only memory of the pre-Reset stress level.
        """
        V_lp0 = self.schedule.v_lp(0.0)
        if V_init is None:
            if V_lp0 <= 0.0:
                raise ValueError(
                    "Cannot infer an initial V from V_lp(0) = 0 (Hold). "
                    "Pass V_init explicitly when starting from a Hold.")
            V_init = V_lp0 * V_fraction
        if V_init <= 0.0:
            raise ValueError("V_init must be strictly positive.")
        if theta_init is None:
            theta_init = self.p.Dc / V_init      # steady state for this V
        if theta_init <= 0.0:
            raise ValueError("theta_init must be strictly positive.")

        tau_0 = self.p.sigma_n * self.friction(V_init, theta_init) \
            + self.p.eta * V_init
        e0 = tau_0 / self.p.k
        return SimulationState(t=0.0, lnV=float(np.log(V_init)),
                               theta=float(theta_init), delta=0.0,
                               u_lp=0.0, e0=float(e0))

    def set_schedule(self, schedule: LoadingSchedule) -> None:
        """Replace the loading schedule WITHOUT touching the physical state.

        The swap takes effect at the next call to advance(), which starts a
        fresh integration from the current state -- i.e. exactly at a chunk
        boundary, never in the middle of a solver step. lnV, theta, delta,
        u_lp and e0 are all left alone, so the spring extension and stress
        are continuous across the change: only the rate at which the load
        point is driven from here on is different.

        This is the ONLY supported way to change loading during a run. The
        frictional parameters (a, b, Dc, sigma_n, V0, G, cs, k) and the
        evolution law are NOT changeable this way, because changing them
        mid-trajectory would discontinuously change mu, eta, kc or the spring
        preload. Those require a Reset.
        """
        if not isinstance(schedule, LoadingSchedule):
            raise TypeError("set_schedule expects a LoadingSchedule instance.")
        schedule.v_lp(0.0)          # provokes the negative-V_lp check now
        self.schedule = schedule
        self._vlp = schedule.v_lp

    # ---- constitutive helpers --------------------------------------------
    def friction(self, V: float | np.ndarray, theta: float | np.ndarray):
        """mu = mu0 + a ln(V/V0) + b ln(V0 theta/Dc)   [-]"""
        p = self.p
        return (p.mu0 + p.a * np.log(V / p.V0)
                + p.b * np.log(p.V0 * theta / p.Dc))

    def _dtheta_dt(self, V, theta):
        """dtheta/dt for the linear-theta formulation. Units: [-] (s/s)."""
        p = self.p
        if self.law == "ageing":
            return 1.0 - V * theta / p.Dc
        x = V * theta / p.Dc
        # x -> 0 gives x ln x -> 0; the limit is taken explicitly rather than
        # by clipping, so the behaviour is documented rather than hidden.
        return np.where(x > 0.0, -x * np.log(np.where(x > 0.0, x, 1.0)), 0.0)

    # ---- right-hand sides -------------------------------------------------
    def _rhs_lnV_theta(self, t, y):
        p = self.p
        L, theta, _, _ = y
        V = np.exp(L)
        dth = self._dtheta_dt(V, theta)
        V_lp = self._vlp(t)
        num = p.k * (V_lp - V) - p.sigma_n * p.b * dth / theta
        den = p.sigma_n * p.a + p.eta * V           # no 1/V anywhere
        return [num / den, dth, V, V_lp]

    def _jac_lnV_theta(self, t, y):
        p = self.p
        L, theta, _, _ = y
        V = np.exp(L)
        V_lp = self._vlp(t)
        den = p.sigma_n * p.a + p.eta * V
        dden_dL = p.eta * V

        if self.law == "ageing":
            dth = 1.0 - V * theta / p.Dc
            dth_dL = -V * theta / p.Dc
            dth_dth = -V / p.Dc
            # d/dtheta of (dth/theta) where dth/theta = 1/theta - V/Dc
            dratio_dth = -1.0 / theta ** 2
            dratio_dL = dth_dL / theta
        else:
            x = V * theta / p.Dc
            lx = np.log(x) if x > 0 else 0.0
            dth = -x * lx
            dth_dL = -x * (lx + 1.0)
            dth_dth = -(x / theta) * (lx + 1.0)
            # dth/theta = -x ln x / theta ; d/dtheta = -x/theta^2
            dratio_dth = -x / theta ** 2
            dratio_dL = dth_dL / theta

        num = p.k * (V_lp - V) - p.sigma_n * p.b * dth / theta
        dnum_dL = -p.k * V - p.sigma_n * p.b * dratio_dL
        dnum_dth = -p.sigma_n * p.b * dratio_dth

        J = np.zeros((4, 4))
        J[0, 0] = (dnum_dL * den - num * dden_dL) / den ** 2
        J[0, 1] = dnum_dth / den
        J[1, 0] = dth_dL
        J[1, 1] = dth_dth
        J[2, 0] = V
        return J

    def _rhs_lnV_lntheta(self, t, y):
        p = self.p
        L, M, _, _ = y
        V = np.exp(L)
        theta = np.exp(M)
        if self.law == "ageing":
            dM = np.exp(-M) - V / p.Dc                # = 1/theta - V/Dc
        else:
            dM = -(V / p.Dc) * (L + M - np.log(p.Dc))  # = -(V/Dc) ln(V theta/Dc)
        V_lp = self._vlp(t)
        num = p.k * (V_lp - V) - p.sigma_n * p.b * dM
        den = p.sigma_n * p.a + p.eta * V
        return [num / den, dM, V, V_lp]

    def _jac_lnV_lntheta(self, t, y):
        p = self.p
        L, M, _, _ = y
        V = np.exp(L)
        V_lp = self._vlp(t)
        den = p.sigma_n * p.a + p.eta * V
        dden_dL = p.eta * V

        if self.law == "ageing":
            dM = np.exp(-M) - V / p.Dc
            dM_dL = -V / p.Dc
            dM_dM = -np.exp(-M)
        else:
            q = L + M - np.log(p.Dc)          # = ln(V theta / Dc)
            dM = -(V / p.Dc) * q
            dM_dL = -(V / p.Dc) * (q + 1.0)
            dM_dM = -V / p.Dc

        num = p.k * (V_lp - V) - p.sigma_n * p.b * dM
        dnum_dL = -p.k * V - p.sigma_n * p.b * dM_dL
        dnum_dM = -p.sigma_n * p.b * dM_dM

        J = np.zeros((4, 4))
        J[0, 0] = (dnum_dL * den - num * dden_dL) / den ** 2
        J[0, 1] = dnum_dM / den
        J[1, 0] = dM_dL
        J[1, 1] = dM_dM
        J[2, 0] = V
        return J

    # ---- state <-> vector -------------------------------------------------
    def _to_vector(self, s: SimulationState) -> np.ndarray:
        if self.formulation == "lnV_theta":
            return np.array([s.lnV, s.theta, s.delta, s.u_lp], dtype=float)
        return np.array([s.lnV, np.log(s.theta), s.delta, s.u_lp], dtype=float)

    def _theta_from_vector(self, y2: np.ndarray | float):
        return y2 if self.formulation == "lnV_theta" else np.exp(y2)

    def _atol(self) -> np.ndarray:
        second = self.atol_state if self.formulation == "lnV_theta" \
            else self.atol_lnV
        return np.array([self.atol_lnV, second, self.atol_disp, self.atol_disp])

    @property
    def _rhs(self) -> Callable:
        return (self._rhs_lnV_theta if self.formulation == "lnV_theta"
                else self._rhs_lnV_lntheta)

    @property
    def _jac(self) -> Callable:
        return (self._jac_lnV_theta if self.formulation == "lnV_theta"
                else self._jac_lnV_lntheta)

    # ---- derived quantities ----------------------------------------------
    def derived(self, t, lnV, theta, delta, u_lp, e0,
                V_lp: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Every quantity the interface displays, from the SAME solution.

        Returned keys (all SI):
            t [s], V_lp [m/s], V [m/s], theta [s], delta [m], u_lp [m],
            e [m] spring extension, mu [-],
            tau_friction [Pa] = sigma_n*mu,
            tau_damping  [Pa] = eta*V,
            tau_total    [Pa] = tau_friction + tau_damping,
            tau_spring   [Pa] = k*e,
            residual     [Pa] = tau_spring - tau_total   (should be ~0),
            V_theta_over_Dc [-]
        """
        p = self.p
        t = np.atleast_1d(np.asarray(t, dtype=float))
        lnV = np.atleast_1d(np.asarray(lnV, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        delta = np.atleast_1d(np.asarray(delta, dtype=float))
        u_lp = np.atleast_1d(np.asarray(u_lp, dtype=float))

        V = np.exp(lnV)
        if V_lp is None:
            V_lp = np.array([self.schedule.v_lp(float(ti)) for ti in t])
        else:
            V_lp = np.atleast_1d(np.asarray(V_lp, dtype=float))
        mu = self.friction(V, theta)
        tau_friction = p.sigma_n * mu
        tau_damping = p.eta * V
        tau_total = tau_friction + tau_damping
        e = e0 + u_lp - delta
        tau_spring = p.k * e
        return dict(
            t=t, V_lp=V_lp, V=V, theta=theta, delta=delta, u_lp=u_lp, e=e,
            mu=mu, tau_friction=tau_friction, tau_damping=tau_damping,
            tau_total=tau_total, tau_spring=tau_spring,
            residual=tau_spring - tau_total,
            V_theta_over_Dc=V * theta / p.Dc,
        )

    # ---- the stepper ------------------------------------------------------
    def advance(self, state: SimulationState, dt_chunk: float
                ) -> Tuple[SimulationState, Dict[str, np.ndarray], StepReport]:
        """Advance the solution by dt_chunk of SIMULATION time.

        The chunk is split at every schedule breakpoint so the integrator
        never steps across an unknown V_lp jump. solve_ivp success is checked
        after EVERY sub-interval; on failure the state is returned unchanged
        from the last good point and success=False is reported.

        Returns (new_state, samples, report). `samples` is the derived-quantity
        dictionary for the newly computed points (excluding the starting point
        to avoid duplicates when appending to a history).
        """
        t0 = state.t
        t1 = t0 + dt_chunk
        if dt_chunk <= 0:
            raise ValueError("dt_chunk must be positive.")

        bps = sorted(set(self.schedule.breakpoints(t0, t1)))
        edges = [t0] + bps + [t1]

        wall0 = time.perf_counter()
        y = self._to_vector(state)
        warnings: List[str] = []
        nfev = njev = nlu = 0
        chunks: List[np.ndarray] = []
        chunk_t: List[np.ndarray] = []
        chunk_vlp: List[np.ndarray] = []
        success, message = True, ""
        t_reached = t0

        sched_max = self.schedule.suggested_max_step()
        for i in range(len(edges) - 1):
            ta, tb = edges[i], edges[i + 1]
            if tb <= ta:
                continue
            max_step = (tb - ta) / self.min_samples_per_chunk
            if sched_max is not None:
                max_step = min(max_step, sched_max)

            # Bind the schedule BRANCH for this sub-interval, so that the
            # collocation stages -- including the one exactly at tb -- all
            # use the V_lp belonging to this sub-interval.
            self._vlp = self.schedule.branch(ta, tb)
            try:
                sol = solve_ivp(self._rhs, (ta, tb), y, method=self.method,
                                jac=self._jac, rtol=self.rtol,
                                atol=self._atol(), max_step=max_step,
                                dense_output=False)
            except Exception as exc:                       # noqa: BLE001
                success = False
                message = (f"solve_ivp RAISED on sub-interval "
                           f"[{ta:.6g}, {tb:.6g}] s: "
                           f"{type(exc).__name__}: {exc}")
                break
            finally:
                self._vlp = self.schedule.v_lp
            nfev += int(getattr(sol, "nfev", 0))
            njev += int(getattr(sol, "njev", 0))
            nlu += int(getattr(sol, "nlu", 0))

            if not sol.success:
                success = False
                message = (f"solve_ivp FAILED on sub-interval "
                           f"[{ta:.6g}, {tb:.6g}] s: {sol.message}")
                break
            if sol.y.size == 0 or not np.all(np.isfinite(sol.y[:, -1])):
                success = False
                message = (f"Non-finite state produced on sub-interval "
                           f"[{ta:.6g}, {tb:.6g}] s. The solution is not "
                           "usable; the physics or the parameters need "
                           "attention, not a larger tolerance.")
                break

            keep = slice(1, None) if sol.t.size > 1 else slice(0, 0)
            chunk_t.append(sol.t[keep])
            chunks.append(sol.y[:, keep])
            branch_fn = self.schedule.branch(ta, tb)
            chunk_vlp.append(np.array([branch_fn(float(ti))
                                       for ti in sol.t[keep]]))
            y = sol.y[:, -1]
            t_reached = float(sol.t[-1])

        if chunk_t and any(c.size for c in chunk_t):
            t_all = np.concatenate([c for c in chunk_t if c.size])
            y_all = np.concatenate([c for c in chunks if c.size], axis=1)
            vlp_all = np.concatenate([c for c in chunk_vlp if c.size])
        else:
            t_all = np.empty(0)
            y_all = np.empty((4, 0))
            vlp_all = np.empty(0)

        theta_all = self._theta_from_vector(y_all[1]) if y_all.size \
            else np.empty(0)
        samples = self.derived(t_all, y_all[0] if y_all.size else np.empty(0),
                               theta_all,
                               y_all[2] if y_all.size else np.empty(0),
                               y_all[3] if y_all.size else np.empty(0),
                               state.e0, V_lp=vlp_all)

        # --- diagnostics ----------------------------------------------------
        if samples["t"].size:
            res = np.abs(samples["residual"])
            max_res = float(res.max())
            min_V = float(samples["V"].min())
            max_V = float(samples["V"].max())
            min_theta = float(samples["theta"].min())
        else:
            max_res, min_V, max_V, min_theta = 0.0, np.nan, np.nan, np.nan

        rel_res = max_res / self.p.sigma_n
        if rel_res > 1e-6:
            warnings.append(
                f"Spring-stress residual reached {max_res:.3e} Pa "
                f"({rel_res:.2e} of sigma_n). Analytically it is zero, so "
                "this is integration error. Tighten rtol/atol or reduce "
                "dt_chunk before trusting the output.")
        if samples["t"].size and min_theta <= 0.0:
            warnings.append(
                f"theta reached {min_theta:.3e} s (non-positive). The "
                "constitutive law's logarithm is undefined there.")
        if samples["t"].size:
            mu_min = float(samples["mu"].min())
            if mu_min <= 0.0:
                warnings.append(
                    f"mu reached {mu_min:.4f} <= 0. The unregularised "
                    "logarithmic RSF law permits negative friction at very "
                    "low V; this is a known limitation of the law as "
                    "implemented, not a numerical fault. A regularised "
                    "(arcsinh) form would be needed to avoid it.")

        if success:
            new_state = SimulationState(
                t=t1, lnV=float(y[0]),
                theta=float(self._theta_from_vector(y[1])),
                delta=float(y[2]), u_lp=float(y[3]), e0=state.e0)
        else:
            # Return the last GOOD state; never fabricate a continuation.
            if samples["t"].size:
                new_state = SimulationState(
                    t=t_reached, lnV=float(y_all[0, -1]),
                    theta=float(theta_all[-1]), delta=float(y_all[2, -1]),
                    u_lp=float(y_all[3, -1]), e0=state.e0)
            else:
                new_state = state

        report = StepReport(
            success=success, message=message, t_start=t0,
            t_end=new_state.t, n_subintervals=len(edges) - 1,
            n_samples=int(samples["t"].size), nfev=nfev, njev=njev, nlu=nlu,
            wall_time_s=time.perf_counter() - wall0,
            max_abs_residual_Pa=max_res, max_rel_residual=rel_res,
            min_V=min_V, max_V=max_V, min_theta=min_theta, warnings=warnings)
        return new_state, samples, report

    # ---- convenience ------------------------------------------------------
    def run(self, state: SimulationState, t_total: float, dt_chunk: float
            ) -> Tuple[SimulationState, "History", List[StepReport]]:
        """Chunked run collecting a History. Stops on the first failure."""
        hist = History()
        hist.append_initial(self, state)
        reports: List[StepReport] = []
        n = int(np.ceil(t_total / dt_chunk))
        for _ in range(n):
            state, samples, rep = self.advance(state, dt_chunk)
            hist.append(samples)
            reports.append(rep)
            if not rep.success:
                break
        return state, hist, reports


# =============================================================================
# 5. LINEAR STABILITY OF THE IMPLEMENTED SYSTEM
# =============================================================================

def stability_jacobian(params: RSFParams, V_ss: float) -> np.ndarray:
    """Reduced 2x2 Jacobian in perturbation coordinates (v, g) where
    V = V_ss(1+v), theta = theta_ss(1+g), theta_ss = Dc/V_ss.

    Identical for the ageing and slip laws (see module docstring).
    Rows/columns: [dv/dt; dg/dt] x [v, g].  Units: 1/s.
    """
    p = params
    s = V_ss / p.Dc                      # 1/s
    A = p.sigma_n * p.a + p.eta * V_ss   # Pa
    return np.array([
        [(-p.k * V_ss + p.sigma_n * p.b * s) / A, p.sigma_n * p.b * s / A],
        [-s, -s],
    ])


def stability_eigenvalues(params: RSFParams, V_ss: float) -> np.ndarray:
    return np.linalg.eigvals(stability_jacobian(params, V_ss))


def stability_summary(params: RSFParams, V_ss: float) -> Dict[str, float]:
    J = stability_jacobian(params, V_ss)
    ev = np.linalg.eigvals(J)
    growth = float(np.max(ev.real))
    return dict(
        trace=float(np.trace(J)), det=float(np.linalg.det(J)),
        growth_rate=growth,                       # Re(lambda)_max [1/s]
        osc_rate=float(np.max(np.abs(ev.imag))),  # |Im(lambda)|   [1/s]
        kc_qs=params.kc_qs, kc_qd=params.kc_qd(V_ss),
        k_over_kc_qs=params.stiffness_ratio_qs(),
        k_over_kc_qd=params.stiffness_ratio_qd(V_ss),
        predicted_unstable=float(growth > 0.0),
    )


# =============================================================================
# 6. INCREMENTAL EVENT DETECTION
# =============================================================================

@dataclass
class Event:
    """One threshold-defined slip event.

    THRESHOLD CHOICE IS AN OPERATIONAL CONVENTION, NOT PHYSICS. An "event" is
    an excursion of V above V_threshold. Changing the threshold changes the
    catalogue. Column names state explicitly WHICH stress is dropping.
    """
    index: int
    t_start: float                # [s]
    t_peak: float                 # [s]
    t_end: float                  # [s]
    duration_s: float             # [s]
    V_max: float                  # [m/s]
    slip_m: float                 # [m] block slip during the excursion
    drop_tau_total_Pa: float      # [Pa] tau_total at onset minus its minimum
    drop_tau_friction_Pa: float   # [Pa] sigma_n*mu at onset minus its minimum
    drop_tau_damping_Pa: float    # [Pa] eta*V at onset minus its minimum
    tau_total_onset_Pa: float     # [Pa]
    tau_friction_onset_Pa: float  # [Pa]
    V_threshold: float            # [m/s] the convention used
    is_possible_spin_up: bool     # first event may reflect the initial condition
    truncated: bool               # still in progress when the run ended


class EventDetector:
    """Stateful, chunk-safe detector. Feed it every chunk of samples in order.

    An event straddling a chunk boundary is handled correctly because the
    accumulator persists between calls.
    """

    def __init__(self, V_threshold: float = DEFAULT_EVENT_THRESHOLD):
        if V_threshold <= 0:
            raise ValueError("V_threshold must be positive.")
        self.V_threshold = float(V_threshold)
        self.events: List[Event] = []
        self._in_event = False
        self._acc: Dict[str, List[float]] = {}
        self._prev: Optional[Dict[str, float]] = None  # last sub-threshold pt

    def _start(self, pre: Optional[Dict[str, float]]):
        self._in_event = True
        self._acc = dict(t=[], V=[], delta=[], tau_total=[], tau_friction=[],
                         tau_damping=[])
        if pre is not None:
            for kk in self._acc:
                self._acc[kk].append(pre[kk])

    def _push(self, row: Dict[str, float]):
        for kk in self._acc:
            self._acc[kk].append(row[kk])

    def _close(self, truncated: bool = False):
        a = {kk: np.asarray(v) for kk, v in self._acc.items()}
        i_peak = int(np.argmax(a["V"]))
        ev = Event(
            index=len(self.events),
            t_start=float(a["t"][0]), t_peak=float(a["t"][i_peak]),
            t_end=float(a["t"][-1]),
            duration_s=float(a["t"][-1] - a["t"][0]),
            V_max=float(a["V"].max()),
            slip_m=float(a["delta"][-1] - a["delta"][0]),
            drop_tau_total_Pa=float(a["tau_total"][0] - a["tau_total"].min()),
            drop_tau_friction_Pa=float(
                a["tau_friction"][0] - a["tau_friction"].min()),
            drop_tau_damping_Pa=float(
                a["tau_damping"][0] - a["tau_damping"].min()),
            tau_total_onset_Pa=float(a["tau_total"][0]),
            tau_friction_onset_Pa=float(a["tau_friction"][0]),
            V_threshold=self.V_threshold,
            is_possible_spin_up=(len(self.events) == 0),
            truncated=truncated)
        self.events.append(ev)
        self._in_event = False
        self._acc = {}

    def feed(self, samples: Dict[str, np.ndarray]) -> List[Event]:
        """Consume one chunk; return events COMPLETED during this chunk."""
        before = len(self.events)
        n = int(samples["t"].size)
        for i in range(n):
            row = {kk: float(samples[kk][i]) for kk in
                   ("t", "V", "delta", "tau_total", "tau_friction",
                    "tau_damping")}
            above = row["V"] > self.V_threshold
            if above and not self._in_event:
                self._start(self._prev)
                self._push(row)
            elif above:
                self._push(row)
            else:
                if self._in_event:
                    self._push(row)     # include the first sub-threshold point
                    self._close()
                self._prev = row
        return self.events[before:]

    def finalise(self) -> None:
        """Close any event still open at the end of a run (marked truncated)."""
        if self._in_event:
            self._close(truncated=True)

    # ---- statistics -------------------------------------------------------
    def recurrence_intervals(self, exclude_spin_up: bool = True) -> np.ndarray:
        """Intervals between successive event onsets [s].

        With exclude_spin_up=True the first event is dropped entirely, because
        it may reflect the initial condition rather than the established limit
        cycle. That is a CHOICE, and it is reported."""
        ev = [e for e in self.events if not e.truncated]
        if exclude_spin_up:
            ev = [e for e in ev if not e.is_possible_spin_up]
        if len(ev) < 2:
            return np.empty(0)
        return np.diff([e.t_start for e in ev])

    def to_rows(self) -> List[Dict]:
        return [vars(e) for e in self.events]

    def to_csv(self, path: str) -> None:
        rows = self.to_rows()
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# no events detected at V_threshold = "
                        f"{self.V_threshold:.6e} m/s\n")
            return
        cols = list(rows[0].keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Event catalogue. V_threshold is an OPERATIONAL "
                    "CONVENTION, not a physical constant.\n")
            f.write("# Stress drops are given separately for total, "
                    "frictional and radiation-damping components.\n")
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")


# =============================================================================
# 7. HISTORY  (structured, suitable for animation and timeline scrubbing)
# =============================================================================

FIELDS = ("t", "V_lp", "V", "theta", "delta", "u_lp", "e", "mu",
          "tau_friction", "tau_damping", "tau_total", "tau_spring",
          "residual", "V_theta_over_Dc")

UNITS = {"t": "s", "V_lp": "m/s", "V": "m/s", "theta": "s", "delta": "m",
         "u_lp": "m", "e": "m", "mu": "-", "tau_friction": "Pa",
         "tau_damping": "Pa", "tau_total": "Pa", "tau_spring": "Pa",
         "residual": "Pa", "V_theta_over_Dc": "-"}


class History:
    """Append-only store of the simulation, in the SAME arrays that the
    animation, the live values, the plots and the CSV export must all use.

    `max_samples` caps memory. When the cap is hit the OLDEST samples are
    dropped and `dropped` records how many, so the interface can say so
    explicitly rather than silently losing data.
    """

    def __init__(self, max_samples: int = 2_000_000):
        self._cols: Dict[str, List[np.ndarray]] = {f: [] for f in FIELDS}
        self._n = 0
        self.max_samples = max_samples
        self.dropped = 0

    def append_initial(self, engine: "SpringSliderEngine",
                       state: SimulationState) -> None:
        s = engine.derived(state.t, state.lnV, state.theta, state.delta,
                           state.u_lp, state.e0)
        self.append(s)

    def append(self, samples: Dict[str, np.ndarray]) -> None:
        n = int(np.asarray(samples["t"]).size)
        if n == 0:
            return
        for f in FIELDS:
            self._cols[f].append(np.asarray(samples[f], dtype=float).ravel())
        self._n += n
        if self._n > self.max_samples:
            self._compact()
            excess = self._n - self.max_samples
            if excess > 0:
                for f in FIELDS:
                    self._cols[f] = [self._cols[f][0][excess:]]
                self._n -= excess
                self.dropped += excess

    def _compact(self) -> None:
        for f in FIELDS:
            if len(self._cols[f]) > 1:
                self._cols[f] = [np.concatenate(self._cols[f])]

    def __len__(self) -> int:
        return self._n

    def arrays(self) -> Dict[str, np.ndarray]:
        self._compact()
        return {f: (self._cols[f][0] if self._cols[f] else np.empty(0))
                for f in FIELDS}

    def index_at_time(self, t: float) -> int:
        """Nearest sample index to time t. This is what timeline scrubbing
        uses to reconstruct the spring-slider picture at an earlier time --
        from stored solution data, never from a re-run."""
        tt = self.arrays()["t"]
        if tt.size == 0:
            return 0
        return int(np.argmin(np.abs(tt - t)))

    def snapshot(self, i: int) -> Dict[str, float]:
        a = self.arrays()
        i = int(np.clip(i, 0, len(self) - 1))
        return {f: float(a[f][i]) for f in FIELDS}

    def decimate(self, n_max: int = 4000) -> Dict[str, np.ndarray]:
        """Even index subsample for PLOTTING ONLY. Never for export or for
        statistics: it will remove the peaks of short events."""
        a = self.arrays()
        n = len(self)
        if n <= n_max:
            return a
        idx = np.unique(np.linspace(0, n - 1, n_max).astype(int))
        return {f: a[f][idx] for f in FIELDS}

    def to_csv(self, path: str, header_note: str = "") -> None:
        a = self.arrays()
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# rsf_engine v{__version__} simulation history. "
                    "STRICT SI UNITS.\n")
            f.write("# Zero-dimensional quasi-dynamic spring-slider. It "
                    "cannot compute nucleation length, rupture-front "
                    "velocity, spatial heterogeneity, edge effects or AE "
                    "locations.\n")
            f.write("# 'residual' = tau_spring - tau_total is analytically "
                    "zero; non-zero values are integration error.\n")
            if header_note:
                for line in header_note.splitlines():
                    f.write(f"# {line}\n")
            f.write(",".join(f"{fl}[{UNITS[fl]}]" for fl in FIELDS) + "\n")
            for i in range(len(self)):
                f.write(",".join(f"{a[fl][i]:.10e}" for fl in FIELDS) + "\n")


# =============================================================================
# 8. PRESETS
# =============================================================================
#
# Presets are defined against the stability of the IMPLEMENTED (quasi-dynamic)
# system, i.e. against kc_QD at the relevant steady velocity, and each is
# checked by computing the linear growth rate rather than by looking for
# events. At laboratory loading rates kc_QD and kc_QS coincide to about 1 part
# in 1e4, so the two references pick nearly the same k; the distinction is
# kept explicit anyway because it is not always small.
# =============================================================================

def preset(name: str, V_lp: float = 1e-6) -> Tuple[RSFParams, LoadingSchedule]:
    """Return (params, schedule) for a named preset.

    'stable_vs'       velocity strengthening (a > b): kc_QS < 0, stable
    'stable_vw'       velocity weakening but k well above kc_QD: stable creep
    'near_critical'   k just above kc_QD: stable, but very slowly decaying
                      oscillations (Hopf threshold nearby)
    'unstable'        k well below kc_QD: stick-slip limit cycle
    'damping_matters' high V_lp, k between kc_QD and kc_QS: the two criteria
                      DISAGREE and the implemented system is stable
    """
    base = dict(mu0=0.60, Dc=20e-6, V0=1e-6, sigma_n=10e6, G=30e9, cs=3000.0)
    if name == "stable_vs":
        p = RSFParams(a=0.010, b=0.005, k=1.0e9, **base)
    elif name == "stable_vw":
        p = RSFParams.from_stiffness_ratio(3.0, "qd", V_ss=V_lp,
                                           a=0.005, b=0.010, **base)
    elif name == "near_critical":
        p = RSFParams.from_stiffness_ratio(1.02, "qd", V_ss=V_lp,
                                           a=0.005, b=0.010, **base)
    elif name == "unstable":
        p = RSFParams.from_stiffness_ratio(0.4, "qd", V_ss=V_lp,
                                           a=0.005, b=0.010, **base)
    elif name == "damping_matters":
        V_lp = 2e-3
        p = RSFParams.from_stiffness_ratio(1.10, "qd", V_ss=V_lp,
                                           a=0.005, b=0.010, **base)
    else:
        raise ValueError(f"unknown preset '{name}'")
    return p, ConstantLoading(V_lp)


PRESET_NAMES = ("stable_vs", "stable_vw", "near_critical", "unstable",
                "damping_matters")


if __name__ == "__main__":
    p, sch = preset("unstable")
    print(f"rsf_engine v{__version__}  --  numerical engine, no interface.")
    print(p.summary(V_ss=sch.steady_V()))
    print()
    for wmsg in p.validate(V_ss=sch.steady_V()):
        print("WARNING:", wmsg)
    print("\nRun test_rsf_engine.py for the Stage-2 validation suite.")
