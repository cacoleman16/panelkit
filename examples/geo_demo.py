"""GeoLift-style geo test design with panelkit — power analysis, guardrails,
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
# 4) Robust DataFrame ingest: messy dtypes are handled.
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
