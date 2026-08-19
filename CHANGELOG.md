# Changelog

All notable changes to **panelkit** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: minor
bumps may add features, patch bumps fix behavior).

The single source of truth for the version is `Cargo.toml [workspace.package]`
(mirrored in `pyproject.toml`); `panelkit.__version__` derives from it.

## [0.3.0] — unreleased

First release since 0.2.8; it collects the feature work that had accumulated on
the `0.2.9` development version without ever being tagged.

### Added
- **MC-NNM**: unpenalized two-way fixed effects with a warm-started lambda path.
- **CP-ASC**: size-controlled conformal inference via a null-imposed full-sample
  refit.
- **SC family**: translation-invariant Augmented SC, plus placebo standard
  errors / confidence intervals reported in ATT units, and the docs-promised
  synthdid jackknife.
- **Callaway–Sant'Anna**: simultaneous sup-t confidence bands, anticipation
  handling, group aggregation, and an event-study plot.
- Top-level `GeoDesign` export (`from panelkit import GeoDesign`).
- Regression tests locking in selected/excluded/eligible-market composition
  across `power()`, `select_markets()`, and `recommend()`.
- **`DemeanedSC`** (Ferman & Pinto 2021): synthetic control fit on unit-demeaned
  data — SC with a free intercept. The answer to a treated market whose *level*
  sits outside the donor hull, where plain SC spends its weights on a gap it
  cannot close and reports the remainder as effect.
- **`RobustSC`** (Amjad, Shah & Shen 2018): hard-thresholds the donor panel's
  spectrum and regresses on the de-noised factors. Weights are unconstrained, so
  it can extrapolate past the donor hull.
- **Penalized SC**: `SyntheticControl(penalty=...)` adds the Abadie & L'Hour
  (2021) dissimilarity penalty, preferring donors that individually resemble the
  treated unit among equally good fits.
- **Donor weight bounds**: `min_weight` / `max_weight` cap or floor every donor's
  weight on top of the simplex, on the estimator classes and on
  `GeoDesign.power()` / `evaluate()` / `select_markets()`. Solved exactly over
  the capped simplex (accelerated projected gradient with a water-filling
  projection and an active-set polish), not clipped after the fact.
- **Configurable geo ensemble**: `ensemble_members=[...]` picks the blend
  independently of which methods are fitted; the ensemble is no longer hard-wired
  to three members.
- `benchmarks/sim_methods.py`: the Monte-Carlo study behind ensemble membership —
  per-method bias/RMSE against a known effect, the error-correlation matrix that
  decides whether a member adds anything, and ensemble RMSE under three weighting
  schemes. `--mde` adds the power-engine view.

### Changed / Fixed
- **Robustness**: the extension never kills the host process — panics unwind into
  a catchable exception, with validation at every Python boundary (so a stray
  panic can't take down a Jupyter kernel).
- **linalg**: fixed scale-dependent thresholds, a proximal-gradient Lipschitz
  bound, and overflow in the Householder/SVD paths.
- **DiD**: drop always-treated units from Sun–Abraham; within-influence-function
  and first-step terms in Callaway–Sant'Anna standard errors; honest handling of
  degenerate cases.
- Fixed a matplotlib `PendingDeprecationWarning` (`Colormap.set_bad` →
  `with_extremes(bad=...)`).
- **Geo defaults**: `power()` and `evaluate()` now fit all five base methods
  (`SC`, `ASC`, `SDID`, `FP`, `RSC`) and blend all of them, so reports gain two
  rows and the `ENSEMBLE` numbers shift. `methods=["SC","ASC","SDID"]` restores
  the previous set. `ensemble_weights` given as a bare list now needs one entry
  per ensemble member (dicts keyed by method name are unaffected).
- **linalg**: the bounded-simplex solver's stopping tolerance is interpreted
  relative to the problem's scale. The Frank–Wolfe duality gap carries the
  objective's units, so an absolute tolerance is unreachable on a panel of
  revenue in the thousands — the solve burned its whole iteration budget and
  still stopped short of the optimum.

### Packaging / CI
- Wheels built for Intel macOS and ARM Linux; wheels + sdist attached to each
  GitHub Release.
- Enriched PyPI classifiers (Development Status, per-minor Python versions
  3.9–3.13, OSI license classifiers, OS Independent).
- Release pipeline hardened: a tag↔version consistency guard refuses to publish
  when the pushed tag disagrees with the manifest version, and a `twine check`
  gate validates the built wheel + sdist before the (immutable) PyPI upload.

## [0.2.1] – [0.2.8] — 2026-06-03/04

Rapid post-launch iteration hardening the estimators and inference:
- Exclude the treated unit from placebo donor pools (no leakage) + a
  single-donor crash guard (0.2.8).
- Calibrated in-space placebo CIs for `evaluate()` (previously
  anti-conservative); a `inference="bootstrap"` option; degenerate-CI guards and
  broad audit-robustness fixes.
- `include`/`exclude` markets for geo design and conservative
  effect-over-time CIs.

## [0.2.0] — 2026-06-03

- Multi-cell geo tests, the SC/ASC/SDID ensemble, and a post-test `evaluate()`
  with a calibrated CI gate.

## [0.1.0] — 2026-06-02

- Initial public release: from-scratch causal-inference estimators for panel /
  geo experiments (SC, ASC, SDID, DiD, MC-NNM) on a dependency-free Rust
  numerical core, exposed to Python via PyO3.

[0.3.0]: https://github.com/cacoleman16/panelkit/compare/v0.2.8...HEAD
[0.2.1]: https://github.com/cacoleman16/panelkit/compare/v0.2.0...v0.2.8
[0.2.0]: https://github.com/cacoleman16/panelkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cacoleman16/panelkit/releases/tag/v0.1.0
