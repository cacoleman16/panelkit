# Benchmarks

All numbers on an Apple-silicon laptop, release build (`maturin develop
--release`), panel size **200 units × 130 periods (104 pre / 26 post)** — the
scale of a realistic geo experiment.

Reproduce with:

```bash
maturin develop --release --manifest-path crates/pypanelkit/Cargo.toml
python benchmarks/bench_sc.py        # single SC fit vs NumPy+SLSQP
python benchmarks/bench_placebo.py   # full placebo inference vs NumPy+SLSQP
cargo bench -p panelkit-estimators   # Rust-side criterion micro-benchmarks
```

## Single synthetic-control fit vs reference

The reference is the textbook implementation: simplex-constrained least squares
solved with `scipy.optimize` SLSQP. panelkit uses its from-scratch away-step
Frank–Wolfe solver in Rust.

| method                    |     ATT |  ms / fit |
|---------------------------|--------:|----------:|
| panelkit (Rust FW)        | 0.05000 |     2.41  |
| reference (NumPy + SLSQP) | 0.05000 |    72.28  |

**≈ 30× faster**, with identical ATT (|Δ| = 3.6e-11).

## Full placebo inference (1 + 199 leave-one-out fits)

Where the per-fit speedup compounds and panelkit's multithreading (rayon) kicks
in. Both compute the same in-space placebo p-value.

| method                       | p-value | seconds |
|------------------------------|--------:|--------:|
| panelkit (Rust, parallel)    |  0.0050 |   0.058 |
| reference (NumPy + SLSQP)    |  0.0050 |  80.29  |

**≈ 1380× faster**, identical p-value.

## Rust-side estimator micro-benchmarks (criterion)

Single fit, 200 × 130:

| estimator | time |
|-----------|-----:|
| SC        | ~2.4 ms |
| ASC       | ~3.3 ms |
| SDID      | ~20 ms |
| MC-NNM    | ~0.5–1.1 s (full SVD per SoftImpute iteration — the intrinsically heavy one) |

## Notes

- Determinism: the multiplier bootstrap and parallel placebo produce
  **bit-identical** output regardless of `RAYON_NUM_THREADS`, because every
  replicate draws from an independent seed-derived substream
  (`Xoshiro256pp::substream(seed, b)`). Verified in CI at 1 and 8 threads.
- The reference numbers exist to anchor the speed comparison against a standard,
  widely-used approach — not to claim the reference is the fastest possible
  Python implementation.
