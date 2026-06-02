# panelkit

Fast, **from-scratch** causal-inference estimators for panel / geo experiments —
written in Rust, exposed to Python.

panelkit reimplements the standard panel causal-inference toolbox on top of its
own dependency-free numerical core (no BLAS/LAPACK, no `ndarray`, no `rand`), so
the whole stack is self-contained, deterministic, and reusable. The numerical
core (`panelkit-linalg`) is a standalone crate intended to also back sibling
projects (e.g. a future time-series library).

## Estimators (v1 scope)

- **Synthetic Control family** — Synthetic Control (Abadie et al. 2010),
  Augmented SC (Ben-Michael et al. 2021), Synthetic DiD (Arkhangelsky et al.
  2021).
- **Difference-in-differences family** — Two-way fixed effects, Callaway &
  Sant'Anna (2021) group-time ATTs, Sun & Abraham (2021) interaction-weighted
  event study.
- **Matrix completion** — MC-NNM via singular-value thresholding (Athey et al.
  2021).

## Inference

Placebo / permutation, jackknife (leave-one-out), block & stationary bootstrap,
multiplier (wild) bootstrap, and conformal block-permutation — all behind a
single `Refittable` contract, all reproducible bit-for-bit regardless of thread
count.

## Architecture

A four-crate Cargo workspace with a strict dependency DAG:

```
linalg  ←  estimators  ←  inference  ←  pypanelkit (PyO3 bindings)
```

Only `pypanelkit` touches Python; `linalg` depends on nothing but `std`.

| crate | role |
|---|---|
| `panelkit-linalg` | Mat type, GEMM/QR/Cholesky, one-sided Jacobi SVD, simplex solvers, SVT, RNG |
| `panelkit-estimators` | the estimators above, as functions of a `Panel` |
| `panelkit-inference` | resampling engines |
| `pypanelkit` | the `panelkit._panelkit` extension module |

## Quick start

```python
import numpy as np
from panelkit import SyntheticControl, SyntheticDiD, CallawaySantAnna

# Y is an N×T outcome array (rows = units, cols = periods).
sc = SyntheticControl(inference="placebo").fit(Y, treated=[0], treat_time=104)
print(sc.att, sc.p_value)

sdid = SyntheticDiD().fit(Y, treated=[0], treat_time=104)        # robust default

# Staggered DiD: per-unit first-treated period, -1/None = never treated.
cs = CallawaySantAnna().fit(Y, treat_start=cohorts)
print(cs.att, cs.event_time, cs.event_att)
```

## Performance

From-scratch Rust + multithreaded inference, on a 200 × 130 panel
(see [BENCHMARKS.md](BENCHMARKS.md)):

| task | panelkit | NumPy + SciPy-SLSQP | speedup |
|------|---------:|--------------------:|--------:|
| single SC fit | 2.4 ms | 72 ms | ~30× |
| full placebo (200 fits) | 0.058 s | 80.3 s | ~1380× |

Identical estimates (ATT |Δ| ≈ 1e-11; same placebo p-value).

## Building from source

```bash
# Rust toolchain (https://rustup.rs) + maturin required.
maturin develop --release --manifest-path crates/pypanelkit/Cargo.toml  # build
cargo test --workspace          # Rust tests (linalg cross-oracle, estimators)
pytest python/tests             # Python-layer tests
```

## License

MIT OR Apache-2.0.
