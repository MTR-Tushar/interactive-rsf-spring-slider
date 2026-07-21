# Interactive Rate-and-State Spring-Slider — v0.2 (Stage 3)

An interactive, PhET-style teaching simulator for a **zero-dimensional,
quasi-dynamic** rate-and-state friction spring-slider.

Prepared for Mohammad Tawhidur Rahman Tushar, incoming DPhil student,
Department of Earth Sciences, University of Oxford.

> **Successful execution is not scientific validation.** That this program
> runs, and that its 107 + 148 automated checks pass, shows only that it
> solves the equations it claims to solve, and that its controls behave as
> specified. It is not evidence that those equations describe any real rock.
> Comparison against published results and laboratory data remains future
> work.

---

## 1. What this is, and what it is not

**It is** the smallest model that produces a complete synthetic laboratory
seismic cycle: interseismic loading → acceleration → instability → rapid slip
→ relaxation → healing → repeat. One block, one spring, one frictional
interface.

**It is not** a spatial fault model. It **cannot** compute:

| Not available | Why |
|---|---|
| nucleation length `Lc` | needs a spatially extended fault |
| rupture-front velocity `Vr` | there is no front; there is no space |
| spatial stress heterogeneity | one block has one stress |
| fault-edge / sample-boundary effects | no geometry exists |
| acoustic emissions or their locations | no micro-scale, no elastodynamics |
| seismograms | radiation damping is **not** full inertia |

The acceleration you see before an event is a **temporal** acceleration, not a
spatial nucleation process.

---

## 2. Installation (Windows)

Open PowerShell or Command Prompt in the project folder, then:

```
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Running

```
python interactive_rsf_spring_slider.py
```

Then open **http://127.0.0.1:8080** in a browser.

To stop it, press `Ctrl+C` in the terminal.

### Running the test suites

```
python test_rsf_engine.py
python test_simulation_controller.py
python benchmark_engine.py
```

---

## 4. Architecture — and the rule it enforces

```
rsf_engine.py            ->  simulation_controller.py  ->  interactive_rsf_spring_slider.py
(physics; interface-free)    (operation; UI-free)          (pixels; physics-free)
```

**The browser never touches the ODE state.** The interface layer contains no
equations. It cannot move the block. A mouse drag can only impose a
*load-point displacement*, which is handed to the controller and integrated by
the engine; where the block ends up is whatever the equations say. If the
picture ever looks wrong, the fix belongs in the engine or the controller.

Every displayed number originates in `rsf_engine.derived()`, reaches the
interface through `SimulationController.snapshot()` / `plot_data()`, and is
only *formatted* by the interface. The animation, the live values and the
graphs are three views of one array, so they cannot disagree.

### Files

| File | Role |
|---|---|
| `rsf_engine.py` | v0.2. Physics. Interface-free, strict SI. |
| `simulation_controller.py` | v0.2. Playback, loading modes, manual pull, operational labels, timeline, threading. |
| `interactive_rsf_spring_slider.py` | NiceGUI front end. No physics. |
| `test_rsf_engine.py` | 107 checks (Stage 2). |
| `test_simulation_controller.py` | 148 checks (Stage 3). |
| `benchmark_engine.py` | Wall-clock cost of `advance()`. |
| `headless_ui_check.py` | Browser-less interface verification. |
| `versions/v0_1/`, `versions/v0_2/` | Frozen snapshots. |

---

## 5. Decision record

### 5.1 Formulation B (`lnV_lntheta`) is the default

Measured on identical 3000 s stick-slip runs:

| formulation | law | nfev | njev | rel. spring residual | θ_min (s) |
|---|---|---|---|---|---|
| `lnV_theta` | ageing | 80 028 | 993 | 9.54e-12 | 2.069e-4 |
| `lnV_theta` | slip | 94 474 | 1106 | 1.87e-11 | 1.925e-4 |
| **`lnV_lntheta`** | ageing | 85 126 | 890 | **7.16e-13** | 2.069e-4 |
| **`lnV_lntheta`** | slip | 94 625 | 982 | **2.94e-13** | 1.925e-4 |

B wins 13×–64× on residual and guarantees θ > 0 by construction, for ~6% more
RHS evaluations. **Caveat recorded:** B's ageing law `e^(-M) - e^L/Dc` suffers
cancellation during rapid slip (both terms ~5e3), which is exactly why
formulation A is **kept and fully tested** as an independent cross-check. The
two agree on recurrence interval to 1 part in 1e8.

### 5.2 `kc_QD`, not `kc_QS`, is the threshold this model obeys

Linearising the **quasi-dynamic** spring-slider about steady sliding gives
`det J = s·k·V_ss/A > 0` always, so stability is decided by the trace alone —
a **Hopf** bifurcation, not a saddle-node. Setting `tr J = 0`:

```
kc_QD = [sigma_n (b - a) - eta V_ss] / Dc          (unstable when k < kc_QD)
```

Both the ageing and the slip law linearise to the same `ġ = -s(v+g)`, which is
why the standard linear threshold does not distinguish them (project text,
Convention 13). `kc_QD < kc_QS` always: **radiation damping is stabilising.**
At `V_lp = 1e-6` the correction is 5 Pa against 5e4 Pa — 1 part in 1e4,
invisible. It only matters at high loading rates, which is what the
`damping_matters` preset is for.

A test at `V_lp = 2e-3`, `k = 1.10 kc_QD` gives `k/kc_QS = 0.88`: **`kc_QS`
predicts instability and is wrong**; the measured growth rate is −3.33333 s⁻¹,
matching `kc_QD`. The mirror case at `0.90 kc_QD` grows at +3.33333 s⁻¹.

### 5.3 NiceGUI, not React or Streamlit

One language; built-in WebSocket; `ui.timer` animation loop. React/FastAPI
would demand a TypeScript build and a protocol design before anything moved.
Streamlit's rerun model is hostile to 25 Hz animation and to dragging. The
migration path to React is preserved because `rsf_engine.py` is interface-free.

---

## 6. Performance, and why a fixed model-time-per-frame was rejected

Measured wall-clock cost of `SpringSliderEngine.advance()` (see
`benchmark_engine.py`):

| chunk | stable median | interseismic median | **rapid-slip median** | **rapid-slip MAX** |
|---|---|---|---|---|
| 0.01 s | 2.42 ms | 1.87 ms | 2.25 ms | **51.3 ms** |
| 0.1 s | 2.22 ms | 2.10 ms | 2.79 ms | **93.2 ms** |
| 1 s | 2.52 ms | 2.32 ms | 2.76 ms | **129.6 ms** |
| 5 s | 2.63 ms | 2.72 ms | 3.10 ms | **181.9 ms** |

Two things follow.

**(a) A worker thread is required.** Medians are 2–3 ms everywhere, but the
rapid-slip tail reaches 182 ms — a five-frame freeze at 30 Hz, occurring
exactly when the user most wants to watch. Integration runs in a daemon
thread (`SimulationWorker`); the interface only reads snapshots. The field
lock is deliberately **released during `solve_ivp`**, verified: 20
`snapshot()` reads completed in 0.8 ms while a 300 ms integration was running.

**(b) A fixed model-time-per-frame cannot work.** The cycle is ~774 s; the
event is ~0.0338 s — a ratio of 2.3e4. Measured: the nucleation ramp from
`V = V_lp` to the 1 mm/s threshold takes **11.7 s**, so one "Fast" frame of
8 s covers 0.7 of the *entire ramp*. With fixed chunks, **0 of 45 UI frames**
sampled `V` above threshold, while History held **451 samples inside the
event** with a largest gap of 1.7e-4 s. The solver was never at fault.

### Chosen strategy — adaptive advancement

Model time per frame is capped by two measured signals:

1. **Slip per frame.** `dt ≤ 0.25 · Dc / max(V, V_lp)`. `Dc` is the
   state-evolution distance, so this guarantees each frame resolves state
   evolution. Interseismically the bound is loose (**5.000 s**); during rapid
   slip it tightens automatically (**6.59e-5 s**).
2. **Log-velocity change per frame.** `dt ≤ 0.35 / |d(lnV)/dt|`. Measured
   `d(lnV)/dt`: **0.0132** s⁻¹ interseismic, **0.125** at `V ~ V_lp`,
   **15.48** at `V ~ 1e-4`, **182.6** in the event — a clean four-decade
   signal, inert during quiet loading and clamping hard on the approach.

Floor `1e-7 s`. Result: **236 UI frames** now sample the event (was 0).
Neither bound touches the equations; they change the chunk size, and Stage 2
measured chunking to be neutral to **1 part in 1e13**.

| setting | value |
|---|---|
| Scene refresh | 25 Hz |
| Plot refresh | 4 Hz |
| Playback speeds | Slow 1, Normal 25, Fast 200 model s per real s |
| Single Step | 0.5 s of model time |

---

## 7. Physical event detection vs visual event display

These are **separate concepts** and are exposed as separate fields.

**Physical event** — owned by the engine's `EventDetector`, based on the
configured threshold and the stored samples. Records onset, end, duration,
peak `V`, peak time, spin-up flag, and the three stress drops separately
(`drop_tau_total`, `drop_tau_friction`, `drop_tau_damping` — never an
unqualified "stress drop"). `snapshot().physical_state_label` is this, and it
is what the scientific record contains.

**Visual latch** — `EVENT_FLASH_HOLD_S = 0.6 s` of **wall-clock** time. An
event lasting 0.0338 s of model time need not coincide with a screen refresh,
so the flash and the "Rapid-slip event" *visual* label are held for a minimum
display time. Exposed as `event_flash_active`, `event_flash_remaining_s`,
`visual_state_label`, `last_visual_event`.

The latch **does not**: alter the trajectory; lengthen the physical event
duration; change the event catalogue; claim `V` is still above threshold; or
alter the stored classification. It is suppressed while scrubbing the
timeline, because a past moment should be labelled by what was happening then.
All six of these are tested separately (test group 17). The Physics tab shows
both labels side by side.

**No dense output is used.** History stores `solve_ivp`'s own adaptive steps
directly, so samples inside an event are solver output, not interpolation
across it — verified: largest stored gap inside the event is 190× smaller than
the event.

---

## 8. Operational state labels

**These are interface conventions, not universal physical phases.** The
project text is explicit that the four-stage decomposition is a
simplification. Every threshold lives in one place —
`StateThresholds` in `simulation_controller.py` — and the Physics tab shows
the exact rule that fired.

| Label | Rule (first match wins) |
|---|---|
| Rapid-slip event | `V > 1e-3 m/s` |
| Post-event relaxation | `< 20 s` since the last event ended |
| Accelerating | `d(lnV)/dt > 1e-3 s⁻¹` **and** `V > 2 V_lp` |
| Stable creep | `0.5 ≤ V/V_lp ≤ 2.0` |
| Nearly locked | `V/V_lp < 0.1` (or `V < 1e-9` under Hold) |
| Transient | none of the above |

The `Accelerating` gate on `V > 2 V_lp` matters: an unstable spring-slider
grows exponentially at `Re λ ≈ 1.5e-2 s⁻¹` throughout its *entire*
interseismic phase, so a bare rate test labelled a block merely settling back
onto steady creep as "Accelerating".

---

## 9. Reset discipline

**Only `V_lp` and the loading mode may change during a run.** Changing `a`,
`b`, `Dc`, `sigma_n`, `V0`, `G`, `cs`, `k` or the evolution law requires
**Reset**, because doing so mid-trajectory would make `mu`, `tau`, `eta`,
`kc` or the spring preload discontinuous. Preset buttons therefore *stage* a
preset — they pause, show the parameter summary, and wait for confirmation.

Changing `V_lp` takes effect at the next chunk boundary and does **not** alter
`V`, `theta`, `delta`, `u_lp`, `e0` or any stress.

---

## 10. Manual pull

Dragging the load-point handle imposes a **piecewise-linear load-point
displacement history**. Per accepted segment: record previous and current
imposed displacement → define the model-time interval → compute that
segment's `V_lp` → integrate the engine → the block moves to wherever the
solution puts it.

* **Forward only.** A backward drag is ignored with a warning; reverse loading
  would need a signed-velocity formulation, which is a different model.
* **If you drag too fast**, the displacement is honoured *exactly* and the
  **model-time interval is stretched** instead, with a saturation warning. The
  alternative — truncating the displacement — would silently disagree with
  where you put the mouse.
* **On release**, the default is **Hold** (`V_lp = 0`).
* The mapping (metres of load-point displacement per screen pixel) is always
  displayed. Sensitivities: low 2e-7, medium 2e-6, high 2e-5 m/px.

---

## 11. Visual magnification

Real displacements are microscopic (a whole cycle advances the load point by
~774 µm; the spring extension varies by ~660 µm). The scene therefore uses a
documented magnification, shown continuously as
**"Visual displacement exaggerated by ×N"**, alongside the actual values in
micrometres.

The magnification adapts to keep motion visible but is passed through a slow
exponential filter, so a scale change can never look like a physical jump.

**Camera convention.** The handle is drawn at a fixed `x` and the ground
pattern scrolls, so the load point's absolute advance is visible as ground
motion while the block's position relative to the handle shows the spring
state. The block's screen offset is `-mag·(e - e0) = mag·(delta - u_lp)` — it
is *derived* from the solution, never animated independently.

---

## 12. Verifying the interface yourself (Windows)

Browser rendering **could not be inspected** in the development sandbox: it
has no chromium, firefox, selenium or playwright. Visual appearance is
therefore **not claimed**. Please check the following locally:

1. Run the app; open http://127.0.0.1:8080.
2. **Explore tab** — press **Play**. Confirm the ground hatching scrolls, the
   spring stretches, the block lags, and the `V_lp` / `V` arrows update.
3. Set speed to **Fast** and wait for an event (~15 s of real time with
   adaptive advancement on). Confirm the block lurches forward, the scene
   flashes and shakes, and the badge reads **Rapid-slip event**.
4. Toggle **Adaptive advancement** off and repeat — the event should now be
   stepped over and invisible, which is the behaviour it exists to prevent.
5. **Graphs tab** — confirm three plots, `V` on a log axis, dashed event
   markers, and that the last plotted point matches the animation.
6. **Physics tab** — confirm the residual is at solver tolerance, the physical
   and visual labels are shown separately, and the limitations block is
   present.
7. Switch to **Manual pull**, drag the handle right; confirm the block does
   *not* follow the pointer. Drag left; confirm the warning. Release; confirm
   Hold.
8. Resize the window and confirm nothing overlaps. **Please report anything
   that does** — this is the one class of defect the automated checks cannot
   see.

---

## 13. Known limitations

### Interface

* Visual appearance unverified in a browser (see §12).
* Scene refresh is a 25 Hz *target*; actual rate depends on your browser and
  the WebSocket round trip. **Smooth real-time animation is not claimed** —
  it has not been observed.
* The SVG scene is re-sent in full each frame (~7 kB). Fine locally; it would
  need diffing over a slow network.
* Timeline scrubbing exists in the controller and is tested, but Stage 3 does
  not yet expose a scrub slider in the UI (Stage 4).
* Parameters other than `V_lp` are displayed but not editable (Stage 4).
* Single browser tab assumed; two tabs share one controller.
* CSV export exists in the engine but is not yet wired to a download button.

### Numerical

* Formulation B's ageing law suffers cancellation during rapid slip; A is
  retained as a cross-check (§5.1).
* The `1 mm/s` event threshold is an **operational convention**. The
  attribution of that value to Romanet et al. (2018) is flagged in the project
  text as **UNVERIFIED**.
* The first event after Reset is flagged `is_possible_spin_up` because the
  initial condition is not on the limit cycle: measured spin-up
  `drop_tau_total` 0.511 MPa vs limit-cycle 0.658 MPa — a 22% difference.
  **Discard the first event** in any analysis.
* Ageing and slip laws share the linear threshold but not the nonlinear
  behaviour: measured recurrence 773.97 s (ageing) vs 677.35 s (slip), 12.5%
  apart.
* Radiation damping bounds `V` but is **not** inertia.
* `solve_ivp` failures pause the run, preserve the partial trajectory and
  report the solver message. Nothing beyond the last good point is invented.

---

## 14. Version history

* **v0.1** — numerical engine + 107-check validation suite. Frozen in
  `versions/v0_1/`, unchanged.
* **v0.2** — Stage 3. Engine additions `set_schedule()` and
  `SimulationState.copy()` (all 107 checks preserved); UI-independent
  controller; adaptive advancement; visual event latch; threading with
  generation IDs; NiceGUI interface; 148-check controller suite.
