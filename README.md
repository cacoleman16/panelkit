# panelkit

Fast, **from-scratch** causal-inference estimators for panel / geo experiments —
written in Rust, exposed to Python.

panelkit reimplements the standard panel causal-inference toolbox on top of its
own dependency-free numerical core (no BLAS/LAPACK, no `ndarray`, no `rand`), so
the whole stack is self-contained, deterministic, and fast. The numerical core
(`panelkit-linalg`) is a standalone crate intended to also back sibling projects
(e.g. a future time-series library).

- **Fast:** ~30× a NumPy+SciPy synthetic control on a single fit, ~1380× on a
  full placebo test (multithreaded). [Details.](BENCHMARKS.md)
- **Self-contained:** the numerical core is hand-written — matmul, Cholesky, QR,
  a one-sided Jacobi SVD, simplex solvers, and a PRNG, with zero numeric deps.
- **Reproducible:** all resampling inference is bit-identical regardless of
  thread count (deterministic per-replicate seed substreams).
- **Modern:** correct under staggered adoption (Callaway-Sant'Anna, Sun-Abraham)
  with a Goodman-Bacon diagnostic, plus a novel conformal-pooled SC family.

## Install

```bash
pip install panelkit            # once published; until then, build from source:

# from a clone (needs a Rust toolchain — https://rustup.rs — and maturin):
pip install maturin numpy
maturin develop --release --manifest-path crates/pypanelkit/Cargo.toml
```

## Data model

Every estimator takes an `N × T` NumPy array `Y` (rows = units, columns = time
periods). Treatment is specified one of two ways:

- **Block treatment** (SC family, MC-NNM, CP-ASC): a list of `treated` unit
  indices and the `treat_time` (first post-treatment column).
- **Staggered adoption** (DiD family): a per-unit `treat_start` array giving each
  unit's first-treated period; use `-1` or `None` for never-treated units.

## Estimators at a glance

| class | method | treatment | best for |
|---|---|---|---|
| `SyntheticControl` | Abadie et al. 2010 | block | one/few treated units, transparent weights |
| `AugmentedSC` | Ben-Michael et al. 2021 | block | poor pre-fit (ridge bias correction) |
| `SyntheticDiD` | Arkhangelsky et al. 2021 | block | **robust general default** |
| `MCNNM` | Athey et al. 2021 | block | low-rank structure, many treated cells |
| `CPASC` | novel (this project) | block, multi-treated | conservative pooled inference, cumulative $ lift |
| `TWFE` | two-way FE | staggered | baseline (biased under heterogeneity) |
| `CallawaySantAnna` | Callaway-Sant'Anna 2021 | staggered | **staggered adoption, event study** |
| `SunAbraham` | Sun-Abraham 2021 | staggered | staggered event study (saturated) |
| `GoodmanBacon` | Goodman-Bacon 2021 | staggered | *diagnostic*: why TWFE is biased |

## Examples

### Synthetic Control (+ placebo inference)

```python
import numpy as np
from panelkit import SyntheticControl

# Y: 50 units × 60 periods; unit 0 treated from period 45.
res = SyntheticControl(inference="placebo").fit(Y, treated=[0], treat_time=45)

res.att                 # average post-treatment effect
res.att_path            # per-period effects (length T_post)
res.counterfactual      # synthetic control's predicted path
res.weights             # donor weights (on the simplex)
res.donor_ids           # which units those weights correspond to
res.p_value             # in-space placebo p-value
print(res.summary())
```

### Synthetic DiD — the robust default

```python
from panelkit import SyntheticDiD

res = SyntheticDiD().fit(Y, treated=[0], treat_time=45)
print(res.att)          # unit + time weighted 2×2 DiD
```

### Augmented SC and MC-NNM

```python
from panelkit import AugmentedSC, MCNNM

AugmentedSC().fit(Y, treated=[0], treat_time=45).att          # ridge-corrected SC
MCNNM().fit(Y, treated=[0], treat_time=45).att                # low-rank completion, λ by CV
```

### CP-ASC — conformal pooled SC (multiple treated units)

```python
from panelkit import CPASC

treated = [0, 1, 2, 3, 4, 5]
res = CPASC(mode="mspe").fit(Y, treated, treat_time=22)   # CP-ASC
res.att                 # empirical-Bayes pooled ATT
res.p_value             # conformal block-permutation p-value
res.unit_att            # per-unit effects
res.unit_weight         # inverse-MSPE pooling weights

CPASC(mode="stratified").fit(Y, treated, 22)   # Strat-CP-ASC (size-robust)
CPASC(mode="cumulative").fit(Y, treated, 22)   # C-AS-CP-ASC (total-dollar target)
```

### Difference-in-differences with staggered adoption

```python
from panelkit import TWFE, CallawaySantAnna, SunAbraham, GoodmanBacon

# treat_start[i] = first treated period for unit i, or -1 if never treated.
cs = CallawaySantAnna().fit(Y, treat_start)
cs.att                  # overall ATT (cohort-size weighted)
cs.event_time           # relative event times, e.g. [-5,...,-1, 0, 1,...]
cs.event_att            # event-study coefficients (clean pre-trends + dynamics)
cs.event_se             # influence-function standard errors
print(cs.summary())

sa = SunAbraham().fit(Y, treat_start)           # interaction-weighted event study
twfe = TWFE().fit(Y, treat_start)               # baseline; biased under heterogeneity

# Goodman-Bacon: why TWFE is biased — decompose it into 2×2 comparisons.
bacon = GoodmanBacon().fit(Y, treat_start)
bacon.twfe              # == TWFE coefficient (Σ weightᵢ · estimateᵢ)
bacon.forbidden_weight  # weight on "already-treated as control" comparisons
print(bacon.summary())
```

Runnable scripts live in [`examples/`](examples/): `sc_demo.py`, `did_demo.py`,
`cpasc_demo.py`. See [GUIDE.md](GUIDE.md) for the estimand, assumptions, and
valid inference for each estimator.

## Inference

| engine | use | determinism |
|---|---|---|
| placebo / permutation (in-space) | SC family, small N treated | order-independent |
| jackknife (leave-one-out) | SDID, N treated ≥ 2 | order-independent |
| multiplier (wild) bootstrap | C&S / SA influence functions | seeded substreams |
| conformal block permutation | CP-ASC, single-unit counterfactuals | order-independent |

All bootstrap/permutation engines produce **bit-identical** results regardless
of `RAYON_NUM_THREADS`, because replicate `b` always draws from
`Xoshiro256pp::substream(seed, b)`. Verified in CI at 1 and 8 threads.

## Performance

From-scratch Rust + multithreaded inference, on a 200 × 130 panel
(see [BENCHMARKS.md](BENCHMARKS.md)):

| task | panelkit | NumPy + SciPy-SLSQP | speedup |
|------|---------:|--------------------:|--------:|
| single SC fit | 2.4 ms | 72 ms | ~30× |
| full placebo (200 fits) | 0.058 s | 80.3 s | ~1380× |

Identical estimates (ATT |Δ| ≈ 1e-11; same placebo p-value).

## Architecture

A four-crate Cargo workspace with a strict dependency DAG:

```
linalg  ←  estimators  ←  inference  ←  pypanelkit (PyO3 bindings)
```

Only `pypanelkit` touches Python; `linalg` depends on nothing but `std`.

| crate | role |
|---|---|
| `panelkit-linalg` | `Mat`, GEMM/QR/Cholesky, one-sided Jacobi SVD, simplex solvers, SVT, RNG |
| `panelkit-estimators` | the estimators above, as functions of a `Panel` |
| `panelkit-inference` | resampling engines |
| `pypanelkit` | the `panelkit._panelkit` extension module |

## Development

```bash
cargo test --workspace                       # Rust tests (incl. SVD cross-oracle)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
maturin develop --release --manifest-path crates/pypanelkit/Cargo.toml
pytest python/tests
cargo bench -p panelkit-estimators           # criterion micro-benchmarks
```

## License

MIT OR Apache-2.0.
