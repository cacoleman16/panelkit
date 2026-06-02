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

## Building from source

```bash
# Rust toolchain (https://rustup.rs) + maturin required.
maturin develop --manifest-path crates/pypanelkit/Cargo.toml   # dev build
cargo test --workspace                                          # Rust tests
```

## License

MIT OR Apache-2.0.
