"""GeoLift-style geo test design with panelkit — power analysis, market
selection, a plain-English report, and a professional figure.
"""

import numpy as np

from panelkit.design import GeoDesign

# --- simulate a realistic geo panel: 40 markets × 78 weeks (1.5 yrs) ---------
rng = np.random.default_rng(7)
N, T = 40, 78
names = [f"DMA_{i:02d}" for i in range(N)]

# latent factors + market size + weak weekly seasonality
uf = rng.normal(size=(N, 3))
tf = rng.normal(scale=0.4, size=(T, 3))
size = np.exp(rng.normal(0.0, 0.6, size=N))            # heterogeneous market size
season = 0.06 * np.sin(2 * np.pi * np.arange(T) / 13)  # quarterly-ish cycle
trend = np.cumsum(0.01 * rng.normal(size=T))
Y = np.zeros((N, T))
for i in range(N):
    base = 1000 * size[i]
    for t in range(T):
        Y[i, t] = base * (1 + trend[t] + season[t] + 0.05 * (uf[i] @ tf[t])) \
                  + base * 0.02 * rng.normal()

design = GeoDesign(Y, names=names)

# --- power analysis for a chosen treatment set ------------------------------
rep = design.power(treated=["DMA_03", "DMA_07", "DMA_11"], test_len=8)
print(rep.summary())
rep.plot("assets/geo_design.png")
print("\nwrote assets/geo_design.png")

# --- market selection: which markets give the best-powered design? ----------
print("\n" + "=" * 64)
print("MARKET SELECTION (target lift 5%, up to 3 markets)")
print("=" * 64)
ranked = design.select_markets(test_len=8, target_lift=0.05, max_treated=3,
                               n_candidates=120, top=5)
for i, c in enumerate(ranked, 1):
    mde = f"{100*c['mde_pct']:.1f}%" if c["mde_pct"] is not None else "—"
    print(f"{i}. {', '.join(c['markets']):<28} power={c['power_at_target']:.2f}  "
          f"MDE={mde:>6}  holdout={100*c['holdout_pct']:.1f}%  conf={c['confidence']:.0f}")

# --- recommendations across specifications (length × #geos × alpha) ----------
print()
grid = design.recommend(
    test_lengths=[4, 8, 12],
    n_geos_options=[1, 2, 3],
    target_lift=0.05,
    alphas=[0.05, 0.10],
    n_candidates=30,
)
print(grid.summary())
grid.plot("assets/geo_scenarios.png")
print("\nwrote assets/geo_scenarios.png")

# --- robust DataFrame ingest: messy dtypes are handled --------------------
try:
    import pandas as pd

    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({
                "dma": names[i],
                # dates as strings (parsed to real dates), shuffled order:
                "week": f"2024-W{t:02d}",
                # outcome as strings with stray formatting:
                "sales": f"{Y[i, t]:.2f}",
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # shuffle rows
    d2 = GeoDesign.from_long(df, location="dma", time="week", outcome="sales")
    print(f"\nfrom_long on messy/shuffled string-typed DataFrame → "
          f"{d2.n} markets × {d2.t} periods OK")
except ImportError:
    print("\n(pandas not installed; skipping DataFrame demo)")
