# Benchmarks

Measured on an **Apple M4 Pro (14 cores)**, release build, panel size
**200 units × 130 periods (104 pre / 26 post)** — the scale of a realistic geo
experiment. Absolute times vary by hardware; re-run to regenerate.

Reproduce (regenerates the figures + `assets/bench_results.txt`):

```bash
maturin develop --release --manifest-path crates/pypanelkit/Cargo.toml
python benchmarks/make_plots.py      # the figures used in the README
python benchmarks/bench_sc.py        # single SC fit vs NumPy+SLSQP
python benchmarks/bench_placebo.py   # full placebo inference vs NumPy+SLSQP
cargo bench -p panelkit-estimators   # Rust-side criterion micro-benchmarks
```

## Single synthetic-control fit vs reference

The reference is the textbook implementation: simplex-constrained least squares
solved with `scipy.optimize` SLSQP. panelkit uses its from-scratch away-step
Frank–Wolfe solver in Rust. Median over 5 panels per size:

| N units |  panelkit (ms) | reference (ms) | speedup |
|--------:|---------------:|---------------:|--------:|
|      20 |          0.032 |          1.43  |   ~45×  |
|      50 |          0.089 |          5.43  |   ~61×  |
|     100 |          0.371 |         23.99  |   ~65×  |
|     150 |          0.945 |         68.95  |   ~73×  |
|     200 |          2.040 |        121.40  |   ~60×  |

Estimates are identical (ATT |Δ| ≈ 1e-11). See `assets/bench_scaling.png`.

> **Robustness note.** SciPy SLSQP has occasional convergence cliffs on
> near-collinear donor panels — in this sweep individual panels ranged from tens
> of ms up to **9.5 s** at N=200 when SLSQP iterated to its cap. The table
> reports the **median** (typical case); the *mean* reference time is far worse.
> panelkit's Frank–Wolfe solver has no such cliffs.

## Full placebo inference (1 + 199 leave-one-out fits)

Where the per-fit speedup compounds and panelkit's multithreading (rayon) kicks
in. Both compute the same in-space placebo p-value.

| method                       | p-value | seconds |
|------------------------------|--------:|--------:|
| panelkit (Rust, parallel)    |  0.0050 |   0.056 |
| reference (NumPy + SLSQP)    |  0.0050 |  82.2   |

**≈ 1467× faster**, identical p-value. See `assets/bench_speedup.png`.

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
