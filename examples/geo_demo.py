"""Geo test design with panelkit — power analysis, guardrails,
market selection, specification recommendations, and professional figures.

Two synthetic panels are used:
  • a clean, well-behaved panel for the power report and guardrails (so a healthy
    design is on display), and
  • a heavy-tailed, noisier panel for the specification sweep (so the number of
    treated geos genuinely changes the answer).
"""

import numpy as np

from panelkit.design import GeoDesign


def clean_panel(n=40, t=78, seed=3):
    """Low-noise factor-model panel — synthetic control tracks tightly."""
    rng = np.random.default_rng(seed)
    uf = rng.normal(size=(n, 3))
    tf = rng.normal(scale=0.4, size=(t, 3))
    size = np.exp(rng.normal(0, 0.5, size=n))
    season = 0.05 * np.sin(2 * np.pi * np.arange(t) / 13)
    trend = np.cumsum(0.008 * rng.normal(size=t))
    Y = np.zeros((n, t))
    for i in range(n):
        base = 1000 * size[i]
        for k in range(t):
            Y[i, k] = base * (1 + trend[k] + season[k] + 0.04 * (uf[i] @ tf[k])) \
                      + base * 0.015 * rng.normal()
    return GeoDesign(Y, names=[f"DMA_{i:02d}" for i in range(n)])


def heterogeneous_panel(n=60, t=78, seed=7):
    """Heavy-tailed sizes + idiosyncratic noise — #geos genuinely matters."""
    rng = np.random.default_rng(seed)
    uf = rng.normal(size=(n, 3))
    tf = rng.normal(scale=0.4, size=(t, 3))
    size = np.exp(rng.normal(0, 0.9, size=n))
    season = 0.06 * np.sin(2 * np.pi * np.arange(t) / 13)
    trend = np.cumsum(0.01 * rng.normal(size=t))
    Y = np.zeros((n, t))
    for i in range(n):
        base = 1000 * size[i]
        noise = 0.10 + 0.10 * rng.random()
        for k in range(t):
            Y[i, k] = base * (1 + trend[k] + season[k] + 0.05 * (uf[i] @ tf[k])) \
                      + base * noise * rng.normal()
    return GeoDesign(Y, names=[f"DMA_{i:02d}" for i in range(n)])


# ===========================================================================
# 1) Power analysis + guardrails on a clean panel.
# ===========================================================================
design = clean_panel()
best = design.select_markets(test_len=8, target_lift=0.05, max_treated=4, top=1)[0]
treated = best["markets"]

rep = design.power(treated=treated, test_len=8)
print(rep.summary())
rep.plot("assets/geo_design.png")
print("\nwrote assets/geo_design.png")

guard = design.diagnose(treated=treated, test_len=8)
print("\n" + guard.summary())
guard.plot("assets/geo_guardrails.png")
print("wrote assets/geo_guardrails.png")

# ===========================================================================
# 2) Market selection (clean panel).
# ===========================================================================
print("\n" + "=" * 64)
print("MARKET SELECTION (target lift 5%, up to 3 markets)")
print("=" * 64)
for i, c in enumerate(design.select_markets(test_len=8, target_lift=0.05,
                                            max_treated=3, top=5), 1):
    mde = f"{100*c['mde_pct']:.1f}%" if c["mde_pct"] is not None else "—"
    print(f"{i}. {', '.join(c['markets']):<28} power={c['power_at_target']:.2f}  "
          f"MDE={mde:>6}  holdout={100*c['holdout_pct']:.1f}%  conf={c['confidence']:.0f}")

# ===========================================================================
# 3) Specification recommendations (heterogeneous panel — #geos matters).
# ===========================================================================
print()
grid = heterogeneous_panel().recommend(
    test_lengths=[4, 8, 12],
    n_geos_options=[3, 5, 10, 20],
    target_lift=0.05,
    alphas=[0.05, 0.10],
    n_candidates=30,
)
print(grid.summary())
grid.plot("assets/geo_scenarios.png")
print("\nwrote assets/geo_scenarios.png")

# ===========================================================================
# 4) Multi-cell test: several disjoint treatment cells at once.
# ===========================================================================
# Real tests often run more than one cell simultaneously (e.g. different
# creatives or budgets per region). Each cell is powered against a SHARED donor
# pool that excludes every cell's treated markets — so cells never borrow each
# other as controls.
print()
mc = design.multi_cell(
    cells={
        "West":     ["DMA_00", "DMA_01", "DMA_02"],
        "Midwest":  ["DMA_10", "DMA_11"],
        "Northeast": ["DMA_20", "DMA_21", "DMA_22", "DMA_23"],
    },
    test_len=8,
    alpha=0.10,
)
print(mc.summary())
mc.plot("assets/geo_multicell.png")
print("\nwrote assets/geo_multicell.png")

# ===========================================================================
# 5) Evaluate a test that already ran (post-test measurement).
# ===========================================================================
# `power()` plans a test; `evaluate()` measures one. The power report above
# already includes an ENSEMBLE row — a weighted average of SC + ASC + SDID
# (auto inverse-variance weights). Here we *run* a synthetic test: inject a known
# +6% lift on the treated markets over the last 8 periods, then recover it.
import numpy as _np  # noqa: E402

Y_test = design.Y.copy()
t_start = design.t - 8
treated_ids = [design.names.index(m) for m in treated]
Y_test[treated_ids, t_start:] *= 1.06            # ground-truth +6% lift
post = GeoDesign(Y_test, names=design.names)

ev = post.evaluate(treated=treated, treat_start=t_start, level=0.90)
print("\n" + ev.summary())
print(f"\n(ground truth was +6.0%; ensemble recovered {100*ev.lift:+.2f}%)")
ev.plot("assets/geo_evaluate.png")
print("wrote assets/geo_evaluate.png")
ev.plot_effect_over_time("assets/geo_effect_over_time.png")
print("wrote assets/geo_effect_over_time.png")

# Pin in must-have markets and drop ones you don't trust as controls:
forced = treated[:1]
ranked = design.select_markets(test_len=8, target_lift=0.05, max_treated=3,
                               include=forced, exclude=[design.names[-1]], top=3)
print(f"\nselect_markets(include={forced}, exclude=['{design.names[-1]}']):")
for c in ranked:
    print(f"   {', '.join(c['markets']):<28} score={c['score']:.3f}  "
          f"(forced market present: {forced[0] in c['markets']})")

# ===========================================================================
# 6) Robust DataFrame ingest: messy dtypes are handled.
# ===========================================================================
try:
    import pandas as pd

    Y = design.Y
    rows = [{"dma": design.names[i], "week": f"2024-W{t:02d}", "sales": f"{Y[i, t]:.2f}"}
            for i in range(design.n) for t in range(design.t)]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # shuffle rows
    d2 = GeoDesign.from_long(df, location="dma", time="week", outcome="sales")
    print(f"\nfrom_long on messy/shuffled string-typed DataFrame → "
          f"{d2.n} markets × {d2.t} periods OK")
except ImportError:
    print("\n(pandas not installed; skipping DataFrame demo)")
