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
