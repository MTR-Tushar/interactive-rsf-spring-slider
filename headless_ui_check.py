"""
headless_ui_check.py -- verification of the Stage-3 interface WITHOUT a browser.

=============================================================================
 HONEST SCOPE -- READ THIS FIRST
=============================================================================
 The environment this was developed in has NO browser (no chromium, firefox,
 selenium or playwright). VISUAL APPEARANCE -- layout, overlap, resizing,
 legibility, animation smoothness -- was therefore NOT verified and is NOT
 claimed. The README gives the Windows steps to check that yourself.

 What this script does verify is real, not a mock:

   PART A  The static element tree embedded in the served page, fetched over
           HTTP from the actually-running server. This is the payload a
           browser receives on load. It proves every control, label, tab,
           preset and limitation text is present.

   PART B  The scene generator and the plot payload, driven directly with a
           real SimulationController snapshot. It proves the SVG is built
           from the numerical solution and contains the required elements.

 Run:  start the app first, in another shell:
           python interactive_rsf_spring_slider.py
       then:
           python headless_ui_check.py
=============================================================================
"""

from __future__ import annotations

import sys
import urllib.request

import numpy as np

import interactive_rsf_spring_slider as APP
import simulation_controller as SC

BASE = "http://127.0.0.1:8080"

# -----------------------------------------------------------------------
# PART A -- the static element tree the browser receives
# -----------------------------------------------------------------------
try:
    html = urllib.request.urlopen(BASE, timeout=15).read().decode()
except Exception as exc:                                        # noqa: BLE001
    print(f"Could not reach {BASE}: {exc}\n"
          f"Start the app first:  python interactive_rsf_spring_slider.py")
    sys.exit(2)


def A(t: str) -> bool:
    return t in html


# -----------------------------------------------------------------------
# PART B -- drive the real scene generator with a real snapshot
# -----------------------------------------------------------------------
ctrl = SC.SimulationController("unstable")
ctrl.play()
for _ in range(60):
    ctrl.advance_frame(1 / 25)
snap = ctrl.snapshot()
svg = APP.build_scene_svg(snap, mag=2.0e5, flash=False, shake=0.0,
                          params=ctrl.params)
svg_flash = APP.build_scene_svg(snap, mag=2.0e5, flash=True, shake=2.0,
                                params=ctrl.params)

# a coseismic snapshot, so the event path is exercised too
ctrl2 = SC.SimulationController("unstable")
ctrl2.set_speed("Fast")
ctrl2.play()
for _ in range(20000):
    if not ctrl2.advance_frame(1 / 25):
        break
    if ctrl2.snapshot().n_events >= 1:
        break
snap2 = ctrl2.snapshot()
svg2 = APP.build_scene_svg(snap2, mag=5.0e4, flash=True, shake=3.0,
                           params=ctrl2.params)
plot = ctrl2.plot_data(n_max=1200)


def B(t: str) -> bool:
    return t in svg or t in svg2


checks = [
    # ---- PART A: static element tree -------------------------------------
    ("A  tabs Explore / Graphs / Physics",
     all(A(t) for t in ("Explore", "Graphs", "Physics"))),
    ("A  playback buttons Play / Pause / Reset / Single Step",
     all(A(t) for t in ("Play", "Pause", "Reset", "Single Step"))),
    ("A  Hold button", A("Hold (V_lp = 0)")),
    ("A  speed toggle Slow / Normal / Fast",
     all(A(t) for t in ("Slow", "Normal", "Fast"))),
    ("A  loading-mode toggle",
     A("Velocity control") and A("Manual pull")),
    ("A  drag-sensitivity control",
     A("Drag sensitivity") and A("medium")),
    ("A  adaptive-advancement switch", A("Adaptive advancement")),
    ("A  all five presets", all(A(t) for t in (
        "Velocity strengthening", "Stable velocity weakening",
        "Near critical", "Stick-slip", "Radiation damping matters"))),
    ("A  live-value rows", all(A(t) for t in (
        "Load-point velocity V_lp", "Block slip velocity V",
        "Spring extension e", "Driving stress tau_total",
        "State variable theta"))),
    ("A  k/kc_QD readout", A("k / kc_QD")),
    ("A  tooltips state provenance",
     A("IMPOSED") and A("INTEGRATED") and A("CALCULATED")),
    ("A  governing equations shown",
     A("d(lnV)/dt") and A("sigma_n mu + eta V")),
    ("A  limitations block", all(A(t) for t in (
        "Zero-dimensional", "Quasi-dynamic", "No nucleation length",
        "No rupture front", "No acoustic emissions",
        "not scientific validation"))),
    ("A  physical-vs-visual label explanation",
     A("PHYSICAL") and A("VISUAL")),
    ("A  three chart elements", html.count("echart") >= 3),
    ("A  interactive image present for dragging",
     "interactive_image" in html.lower()),
    ("A  loading-mode row present", A("Loading mode")),

    # ---- PART B: the generated scene --------------------------------------
    ("B  scene builds and is substantial", len(svg) > 2500),
    ("B  slider block drawn", B("SLIDER BLOCK")),
    ("B  load-point handle drawn", B("LOAD POINT")),
    ("B  frictional interface drawn", B("frictional interface")),
    ("B  spring drawn with its stiffness", B("spring k =")),
    ("B  V_lp arrow labelled", B("V_lp =")),
    ("B  V arrow labelled", B("V =")),
    ("B  driving-stress indicator", B("driving stress")),
    ("B  frictional-resistance indicator", B("frictional resistance")),
    ("B  operational state badge", B(snap.visual_state_label)),
    ("B  magnification notice", B("exaggerated by")),
    ("B  event-flash path renders", 'opacity="0.10"' in svg_flash),
    ("B  scene reflects the numerical solution (mu matches snapshot)",
     f"{snap.mu:.4f}" in svg),
    ("B  no NaN or inf leaked into the SVG",
     "nan" not in svg.lower() and "inf" not in svg2.lower()),

    # ---- PART B: plot payload ---------------------------------------------
    ("B  plot data decimated for display only",
     plot["n_shown"] <= 1200 and plot["n_full"] > plot["n_shown"]),
    ("B  event markers present in plot payload",
     len(plot["event_times"]) >= 1),
    ("B  plot latest point equals the animation's latest time",
     abs(float(plot["t"][-1]) - snap2.t) < 1e-9),
    ("B  all plotted series finite",
     all(bool(np.isfinite(plot[k]).all())
         for k in ("V", "V_lp", "tau_total", "tau_friction", "u_lp",
                   "delta"))),
]

npass = sum(1 for _, ok in checks if ok)
print("=" * 74)
print(" headless_ui_check.py  --  NO BROWSER AVAILABLE; appearance NOT judged")
print("=" * 74)
print(f" static page fetched: {len(html):,} bytes")
print(f" scene generated    : {len(svg):,} chars (interseismic), "
      f"{len(svg2):,} chars (coseismic)")
print(f" plot payload       : {plot['n_full']:,} stored, "
      f"{plot['n_shown']:,} plotted, "
      f"{len(plot['event_times'])} event marker(s)\n")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n {npass}/{len(checks)} headless checks passed")
print("\n NOT VERIFIED HERE: visual layout, overlap, resizing, font "
      "legibility,\n animation smoothness, browser performance. Those "
      "require a browser.")
sys.exit(0 if npass == len(checks) else 1)
