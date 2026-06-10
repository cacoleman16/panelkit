# panelkit methods guide

A practitioner's reference: for each estimator, what it targets (the estimand),
the assumptions it leans on, when to reach for it, and which inference is valid.
All estimators take an `N × T` outcome matrix and return an average treatment
effect on the treated (ATT) — but *which* ATT differs, so read the estimand line
carefully.

---

## Synthetic Control (`SyntheticControl`)

**Estimand.** ATT for the treated unit(s): observed minus a synthetic
counterfactual built as a convex combination of control ("donor") units.

**How.** Choose non-negative donor weights summing to one that best match the
treated unit's *pre-treatment* outcome path (`min ‖Y₁,pre − Y₀,pre w‖²` over the
simplex), then read the effect off the post-period gap. Solved with an away-step
Frank-Wolfe solver (sparse vertex solutions, exact on faces).

**Assumptions.** The treated unit's untreated path lies (approximately) in the
convex hull of donors; good pre-treatment fit; no anticipation; donors
unaffected by the treatment (no spillover/SUTVA violations).

**Use when.** One or a few treated units; you want transparent, interpretable
weights; the donor pool plausibly spans the treated unit.

**Inference.** In-space **placebo / permutation** (`inference="placebo"`):
reassign treatment to each donor, refit, and rank the treated unit's post/pre
RMSPE ratio against the placebo distribution. Valid for small numbers of treated
units. Watch `pre_rmspe` — a poor pre-fit invalidates the comparison.

---

## Augmented Synthetic Control (`AugmentedSC`)

**Estimand.** Same ATT as SC.

**How.** SC plus a ridge **bias correction**: regress donors' post-period
outcomes on their pre-period outcomes (ridge) and adjust the counterfactual by
the pre-period imbalance projected through that map. When SC fits perfectly,
ASC = SC.

**Use when.** No convex combination matches the treated unit well (residual
pre-treatment imbalance) — ASC extrapolates in a controlled, regularized way.

**Inference.** Placebo, as for SC.

---

## Synthetic Difference-in-Differences (`SyntheticDiD`)

**Estimand.** ATT for the treated unit(s).

**How.** Combines SC-style **unit weights** (match the treated pre-period path,
ridge-regularized) with **time weights** (match the post-period across controls),
then forms a doubly-weighted 2×2 difference-in-differences. More robust to
violations of either pure-SC or pure-DiD assumptions than either alone.

**Use when.** The general-purpose default for a block-treatment geo test.

**Inference.** **Jackknife** (leave-one-unit-out) when there are ≥ 2 treated
units; otherwise placebo.

---

## Matrix Completion NNM (`MCNNM`)

**Estimand.** ATT on treated cells: observed minus a low-rank imputed
counterfactual.

**How.** Treat the treated post-period cells as *missing* and complete the
outcome matrix as `L + Γ⊕Δ`: **unpenalized two-way fixed effects** (unit and
time levels, as in the paper) plus a low-rank `L` found by **SoftImpute** —
iterate (re-fit FE on observed cells → fill missing with the current `L` → SVD
→ soft-threshold the singular values by `λ` → reconstruct) to a fixed point.
Penalizing only `L` matters: without the FE terms the outcome *level* itself
gets shrunk by `λ`, biasing every imputed cell. `λ` is chosen by
cross-validation over held-out observed cells, walking the grid from large to
small λ with warm starts (deterministic given the seed).

**Assumptions.** The untreated potential-outcome matrix is approximately
low-rank after removing unit and time levels (a latent-factor structure).

**Use when.** Many treated cells / a genuine factor structure; you do not want to
commit to explicit unit weights. It is the intrinsically heavy estimator (a full
SVD per iteration) — pass `max_rank=k` to switch the inner SVD to a fast,
self-contained **randomized truncated SVD** (a large speedup when the
counterfactual is low-rank, which it usually is).

**Inference.** Conformal / block resampling (the point estimate is the primary
output in v1).

---

## CP-ASC family (`CPASC`) — novel

For experiments with **several treated units**, fit one augmented SC per treated
unit and **pool** the per-unit effects. Conservative by design (near-0%
false-positive rate via conformal inference).

- `mode="mspe"` — **CP-ASC**: empirical-Bayes pooling, weighting unit `d` by
  `1 / (mspe_d + median(mspe))`. Poorly-fitting units are shrunk toward the pool.
- `mode="stratified"` — **Strat-CP-ASC**: stratify units by size (baseline level)
  into bins, MSPE-pool within each bin, average bins by unit count. Protects
  against a single extremal large unit dominating the pool under heterogeneous
  effects.
- `mode="cumulative"` — **C-AS-CP-ASC**: weight by baseline size, targeting the
  baseline-weighted **cumulative** ATT — the quantity that maps to total dollar
  lift rather than the equal-weighted average effect.

**Estimand.** Pooled ATT across treated units (equal-ish for `mspe`/`stratified`,
baseline-weighted/cumulative for `cumulative`).

**Inference.** **Conformal block permutation** on the pooled residual path: under
the sharp null and residual stationarity, the actual post-treatment block is
exchangeable with all circularly-shifted blocks; the p-value is the share at
least as extreme. Tune resolution with `block_len`.

---

## Two-way fixed effects (`TWFE`)

**Estimand.** With a single adoption time and homogeneous effects, the ATT. Under
**staggered adoption with heterogeneous effects, a contaminated weighted
average** that can even be sign-flipped — see Goodman-Bacon below.

**How.** Regress the outcome on a treatment dummy after absorbing unit and time
fixed effects (the two-way within transform). Cluster-robust SE by unit.

**Use when.** A baseline/benchmark. For staggered designs prefer C&S or SA.

---

## Callaway & Sant'Anna (`CallawaySantAnna`)

**Estimand.** Group-time ATTs `ATT(g, t)` — the effect for cohort `g` (units first
treated at period `g`) at period `t` — then clean aggregations: an event-study
path by relative time `e = t − g`, and an overall cohort-size-weighted ATT.

**How.** Each `ATT(g, t)` is a 2×2 long-difference vs the never-treated group
using base period `g − 1`. Every `ATT(g, t)` carries a unit-level **influence
function**, so aggregations get correct standard errors and a multiplier
bootstrap can resample them directly.

**Assumptions.** Parallel trends (conditional on never-treated); no anticipation.
Pre-treatment event-study coefficients (`e < −1`) should be ≈ 0 — a built-in
falsification check.

**Use when.** Staggered adoption; you want an honest event study and an overall
ATT robust to treatment-effect heterogeneity.

**Control group.** `control_group="never"` (never-treated, default) or
`"notyet"` (not-yet-treated: never-treated units plus cohorts treated strictly
after the periods involved). Not-yet-treated gives a larger control pool and
**works even when there are no never-treated units**.

**Covariate adjustment.** Pass `covariates=X` (an `N×K` array of time-invariant
unit characteristics) to use the **regression-adjustment** variant: each
group-time ATT regresses the long-difference on `[1, X]` among the comparison
group and subtracts the fit, removing covariate-driven differential trends. With
no covariates it reduces exactly to the simple estimator.

**Inference.** Influence-function SEs (analytic) or **multiplier (wild)
bootstrap**, clustered by unit. *(With covariates the IF omits the
β-estimation correction term, so those SEs are approximate; a fully
doubly-robust variant — adding an IPW propensity model — is the next step.)*

---

## Sun & Abraham (`SunAbraham`)

**Estimand.** Interaction-weighted event-study coefficients: cohort-specific
effects `CATT(e)` aggregated by cohort share to a clean coefficient per relative
time `e`, plus an overall ATT.

**How.** A saturated two-way FE regression of the outcome on cohort × relative-
time interactions (never-treated as reference, base period `e = −1` omitted),
solved by QR after absorbing unit/time FE. Cluster-robust covariance propagated
to the aggregates.

**Use when.** Staggered event study; a regression-based alternative/robustness
check to C&S.

---

## Goodman-Bacon decomposition (`GoodmanBacon`)

**Not an estimator — a diagnostic.** Decomposes the TWFE coefficient into the
weighted average of every 2×2 comparison: treated-vs-never-treated,
earlier-vs-later (clean), and **later-vs-earlier "forbidden"** comparisons that
use *already-treated* units as controls. `result.twfe` reproduces the TWFE
coefficient exactly; `result.forbidden_weight` quantifies how much of it rests on
the bias-inducing comparisons. Large forbidden weight + heterogeneous effects ⇒
distrust TWFE, use C&S/SA.

---

## Choosing an estimator (rules of thumb)

- **Block treatment, one treated unit:** SC or ASC, placebo inference.
- **Block treatment, general default:** SDID.
- **Block treatment, several treated units, conservative read / cumulative $:**
  CP-ASC family.
- **Low-rank structure / many treated cells:** MC-NNM.
- **Staggered adoption:** C&S (headline) and SA (robustness); run GoodmanBacon to
  show why a naive TWFE differs.

---

## Geo test design (`panelkit.design`)

The planning layer in front of a geo experiment: which markets to treat, how big
a lift you can detect, which test specification to run, and whether to trust the
design. Heavy simulation runs in Rust (the `panelkit-geo` crate); the Python
`GeoDesign` class adds ingest, reports, and figures.

### Loading data

```python
from panelkit.design import GeoDesign

# (a) straight from an N×T matrix (rows = markets, cols = periods, chronological)
design = GeoDesign(Y, names=market_names)

# (b) from a long/tidy DataFrame — robust to messy dtypes, no pre-cleaning needed
design = GeoDesign.from_long(df, location="dma", time="week", outcome="sales")
```

`from_long` is deliberately forgiving of real-world data:

| issue | what `from_long` does |
|---|---|
| outcome stored as strings (`"1234.5"`) | coerced to numeric |
| genuinely non-numeric outcome (`"N/A"`, `"1,234"`) | **errors** with the offending value |
| dates as strings (`"2024-01-07"`) or unsorted | parsed to datetime, columns ordered chronologically |
| non-date periods (ints, labels) | ordered numerically, else lexically |
| location as ints/categoricals | cast to string market names |
| duplicate (location, time) rows | aggregated (`agg="sum"` default) with a warning |
| missing (market × period) cells | **errors**, reporting how many are missing |

The panel must end up **balanced and finite**; the errors tell you exactly what
to fix.

### Power analysis — `design.power(treated, test_len, …)`

Historical placebo with injected multiplicative lift on your real panel, across
SC / ASC / SDID. Returns a report with:

- **MDE three ways**: `% lift`, `absolute` per-period, and `cumulative` over the
  window, each with confidence intervals.
- a **0–100 confidence score**, a one-line **verdict**, and plain-language
  **warnings** (weak fit, volatility, seasonality, holdout, donor count).
- `rep.summary()` (text) and `rep.plot(path)` (3-panel figure: power curves,
  estimate-accuracy CI, design-quality bars).

Key options: `alpha` (significance level, default 0.10), `target_power`
(default 0.80), `lifts` (the % grid), `methods`, `recommended` (default SDID),
`lookback`, `ensemble`/`ensemble_weights`.

**The ENSEMBLE method (weighted average of SC + ASC + SDID).** By default
`power()` adds an `"ENSEMBLE"` result alongside the three base methods: a
weighted average of their ATTs, combined *within each placebo window* before the
null and power are computed. (That ordering matters — the power of the averaged
estimator is not the average of three powers; the blend is usually steadier than
any single method, so its MDE is often the smallest.) `ensemble_weights="auto"`
(default) uses **inverse-variance** weighting — each method weighted by the
precision of its historical-null distribution, so a noisier estimator counts for
less. Pass `"equal"`, a dict like `{"SC": 0.5, "ASC": 0.2, "SDID": 0.3}`, or a
`[w_sc, w_asc, w_sdid]` list to set them yourself; `ensemble=False` turns it off.
The weights used are printed in the report and stored on
`rep.results["ENSEMBLE"].ensemble_weights`.

**How power is simulated (many placebos, not one).** For a treated set, the test
window of length `test_len` is *slid across the whole history*: every valid start
position is one placebo experiment. The detection threshold (critical |ATT|)
comes from those same windows with **no** injected lift (the historical null), and
power at lift τ is the share of windows whose injected effect clears that
threshold. So the estimate is averaged over **many** placebos — `result.n_windows`
reports how many.

**The `lookback` option — how far back to simulate.** By default panelkit powers
over *all* valid windows (more placebo samples → a more stable power estimate).
Pass `lookback=k` to use only the **most-recent k** windows: those have the
longest pre-periods and reflect current dynamics, so they're the most
representative of the test you're about to run — at the cost of fewer samples (a
noisier estimate). It matters when older history is unrepresentative (regime
change, growth, format changes) or when early windows have very short pre-periods;
set `lookback` to cover your relevant recent history (e.g. the last ~6–12
months of windows).

### Evaluating a test that ran — `design.evaluate(treated, treat_start, …)`

`power()` *plans* a test; `evaluate()` *measures* one. Given the treated markets
and the period treatment began (`treat_start`, the first post-period column), it
fits SC / ASC / SDID, reports each one's realized effect, and blends them into a
weighted-average **ensemble** estimate.

```python
ev = design.evaluate(treated=["chicago", "denver"], treat_start=52, level=0.90)
print(ev.summary())               # per-method + ensemble lift, CI, cumulative
ev.plot("evaluate.png")           # observed-vs-cf, effect path (CI band), lift bar
ev.plot_effect_over_time("effect.png")  # pointwise + cumulative over time, w/ CIs
ev.lift, ev.cumulative, ev.significant
```

Inference defaults to **in-space placebo** (Abadie, `inference="placebo"`): every
donor market is refit as if it were the treated one, and the spread of *their*
post-period effects is the null reference — capturing out-of-sample extrapolation
error, the real source of uncertainty. A second engine, `inference="bootstrap"`,
uses a moving-block bootstrap of the pre-period residuals; it's serial-correlation
aware and works as a **fallback when the donor pool is too small for placebo**, but
it only sees in-sample noise, so it is *optimistic* (the report is flagged
`optimistic` and you shouldn't lean on it for significance). (A bootstrap of the treated unit's own post-period only sees
in-sample noise and is wildly anti-conservative — on null data its 90% interval
falsely flags an effect ~50% of the time; the placebo version sits at/below the
nominal 10%.) Poorly-fit placebos (pre-period RMSPE > 2× the treated unit's) are
dropped, per Abadie. The p-value is the placebo rank of the treated effect, and
`"auto"` ensemble weights are inverse-variance from each method's placebo-null
spread. `ev` exposes
`.lift`, `.att`, `.cumulative`, `.significant`, the per-method results in `ev.per`,
and the ensemble in `ev.ensemble`. Reported numbers: **% lift** (effect ÷
counterfactual), **per-period ATT**, and **cumulative incremental** over the
window (summed across treated markets).

**Effect over time** (`ev.plot_effect_over_time(...)`) gives the event-study view:
the **pointwise** effect across the full timeline — *including the pre-period*, so
you can see it sits flat (centered on zero) inside the noise band before the test
starts (a placebo check) and breaks out after — and the running **cumulative
incremental**, each as a point estimate with a confidence band. The counterfactual
is centered on the pre-period, so the gap shows fit quality rather than a level
offset (SDID matches trends, not levels). The bands come from the **in-space
placebo** distribution: at each horizon, the pointwise band is the spread of the
donor placebos' per-period effects, and the cumulative band is the spread of their
cumulative sums (so it fans out with horizon). Placebo inference needs a decent
donor pool to have power — with only a handful of comparable donors the intervals
are necessarily wide. Pass `exclude=[…]` to drop markets from the control pool
(e.g. ones you don't trust as donors).

### Choosing a specification — `design.recommend(test_lengths, n_geos_options, target_lift, alphas=…)`

Sweeps designs across **test length × number of geos × alpha** and recommends the
best (smallest MDE among trustworthy designs, ties broken toward shorter/cheaper).
`grid.summary()` prints the recommendation + alternatives; `grid.plot(path)`
renders the **tradeoffs figure**. Use it to find the "knee" — the cheapest design
that still detects your target lift.

**Reading the tradeoffs figure:**
- **Top panel** — minimum detectable lift (%) vs test length, one line per number
  of treated geos. *Lower is better.* The red band marks lifts you *can't*
  detect; lines below your target lift are viable designs. More geos and longer
  tests pull the line down (more signal), but cost more holdout/time — pick the
  knee where the curve flattens.
- **Bottom-left heatmap** — the same MDE across every (test length × #geos) cell,
  green = small detectable lift (good), red = large (bad), grey = underpowered.
- **Bottom-right** — with multiple alphas, how the MDE of the recommended design
  moves with the significance level (looser α → smaller MDE, more false
  positives); with one alpha, design confidence by spec.
- The black ★ marks the recommended design.

### Guardrails — `design.diagnose(treated, test_len)`

Before trusting a design, check it. `diagnose` returns a report with
`.summary()` and `.plot(path)` (the **guardrails figure**): the pre-period fit
(treated vs synthetic control, so you can *see* whether the counterfactual
tracks), a seasonality ACF, the holdout share against a healthy band, and a
banner listing any plain-language warnings (weak fit, volatile markets, strong
seasonality vs short history, tiny/huge holdout, too few donors). It also exposes
`.confidence`, `.holdout_pct`, and `.warnings`.

### Picking markets — `design.select_markets(test_len, target_lift, max_treated, …)`

Searches candidate treatment-market sets and ranks them by power, MDE, pre-fit,
holdout, and confidence. Pass `eligible=[…]` to restrict to markets you can
actually run in.

Two real-world controls for *which* markets the search may use:

- **`include=[…]`** — force specific markets into **every** candidate treatment
  set (must-treat markets, e.g. a flagship region you've already committed to).
  The search fills the remaining slots from `eligible`, up to `max_treated`.
- **`exclude=[…]`** — drop markets **entirely**: they're never treated *and*
  never used as a donor/control (e.g. a market with contaminated data or its own
  concurrent campaign). `exclude` is also accepted by `power()`, `diagnose()`,
  `evaluate()`, and `recommend()` to keep a market out of the control pool.

### Multi-cell tests — `design.multi_cell(cells, test_len, …)`

Often you run several treatment cells at once — different creatives, budgets, or
messages across disjoint groups of markets — and want each cell's lift measured
separately. The subtlety is the control pool: a market that's treated in one cell
can't be a clean control for another. `multi_cell` handles this by powering each
cell against a **shared donor pool that excludes every cell's treated markets**.

```python
mc = design.multi_cell(
    cells={
        "West":      ["los_angeles", "san_diego"],
        "Midwest":   ["chicago", "detroit"],
        "Northeast": ["boston", "philadelphia"],
    },
    test_len=8, alpha=0.10,
)
print(mc.summary())          # per-cell MDE / confidence / holdout + combined holdout
mc.plot("multicell.png")     # per-cell power curves + an MDE-by-cell bar
```

`cells` maps a label to its markets (names or indices) and must be disjoint. By
default the donor pool is every market not assigned to any cell; pass
`shared_donors=[…]` to fix it explicitly. `lifts`, `methods`, `alpha`,
`target_power`, `recommended`, and `lookback` are forwarded to each cell's power
analysis. The report exposes `mc.cells[label]` (a full power report per cell) and
a combined holdout across all cells. Bigger cells get a smaller MDE; underpowered
cells are flagged so you can grow or merge them before spending.

### What the design layer gives you

Multi-method power (SC/ASC/SDID plus a weighted-average **ensemble** and a
naive-DiD baseline), MDE in %/absolute/cumulative with CIs, an explicit 0–100
confidence score + one-line verdict, seasonality/stability/holdout guardrails with
plain-English warnings, a specification-tradeoff sweep, multi-cell designs,
**post-test evaluation** (`evaluate()`), and publication-clean figures out of the
box.
