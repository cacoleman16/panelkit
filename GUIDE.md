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
outcome matrix with **SoftImpute** — iterate (fill missing with the current
estimate → SVD → soft-threshold the singular values by `λ` → reconstruct) to a
low-rank fixed point. `λ` is chosen by cross-validation over held-out observed
cells (deterministic given the seed).

**Assumptions.** The untreated potential-outcome matrix is approximately
low-rank (a latent-factor structure).

**Use when.** Many treated cells / a genuine factor structure; you do not want to
commit to explicit unit weights. It is the intrinsically heavy estimator (a full
SVD per iteration).

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

**Inference.** Influence-function SEs (analytic) or **multiplier (wild)
bootstrap**, clustered by unit.

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
