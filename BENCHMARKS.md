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

### Per-fit time by estimator (N=200, T=130)

The constrained-weight estimators all beat SLSQP; SDID most (two weight problems
per fit). See `assets/bench_methods.png`.

| estimator | panelkit | reference (SLSQP) | speedup |
|---|---:|---:|---:|
| SC   |  2.07 ms |  125 ms  | ~60×  |
| ASC  |  2.75 ms |  127 ms  | ~46×  |
| SDID | 11.3 ms  | 1576 ms  | ~139× |

### The honest exception: MC-NNM (LAPACK wins)

MC-NNM's inner loop is an SVD, run ~100× by SoftImpute. panelkit's from-scratch
one-sided Jacobi SVD is **~20× slower** than NumPy's LAPACK-backed
`np.linalg.svd`:

| MC-NNM @N=200 (fixed λ) | per fit |
|---|---:|
| panelkit (Jacobi SVD) | ~112 ms |
| reference (LAPACK SVD) | ~5 ms |

This is expected — LAPACK is decades of vendor-tuned assembly. panelkit keeps
MC-NNM **self-contained**, not fastest. If MC-NNM at scale matters, swapping in a
LAPACK/BLAS-backed SVD behind a feature flag is the lever.

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

## Monte Carlo & scale

For power analysis, robustness sweeps, and large simulation studies the workload
is "fit the estimator on thousands of panels." Levers, in order of impact:

1. **`fit_many` (shipped).** Runs the whole replication loop in Rust across all
   cores (rayon), GIL released, returning one ATT per panel. ~2000 SC fits in
   ~0.05 s; a 5-point power curve (12k fits) in ~3.6 s. Avoids Python per-fit
   overhead *and* gives cross-replication parallelism. Prefer this over a Python
   `for` loop or `joblib` (no pickling, no process spawn, shared memory).
2. **Determinism for free.** All resampling uses per-replicate seed substreams,
   so results are reproducible and thread-count-invariant — you can parallelize
   without worrying about RNG ordering, and re-runs are exact.
3. **Cheap inference where possible.** For a power curve you usually only need
   the point estimate per rep (which `fit_many` returns); reserve the full
   placebo/bootstrap for the final reported design, not every MC cell.
4. **Reuse fixed structure (future lever).** When the donor pool is held fixed
   across reps (only the treated series / noise changes), the donor Gram
   `Z₀ᵀZ₀` can be factored once and reused — a further constant-factor win for
   SC/ASC. Not yet exposed as a dedicated API; open an issue if useful.
5. **MC-NNM at scale.** If your sweep leans on MC-NNM, the SVD dominates — a
   LAPACK/BLAS-backed SVD behind a feature flag would be the highest-impact
   change (see the MC-NNM note above).

## Notes

- Determinism: the multiplier bootstrap and parallel placebo produce
  **bit-identical** output regardless of `RAYON_NUM_THREADS`, because every
  replicate draws from an independent seed-derived substream
  (`Xoshiro256pp::substream(seed, b)`). Verified in CI at 1 and 8 threads.
- The reference numbers exist to anchor the speed comparison against a standard,
  widely-used approach — not to claim the reference is the fastest possible
  Python implementation.
