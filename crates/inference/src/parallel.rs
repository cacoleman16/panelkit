//! Parallel resampling driver with **deterministic per-replicate substreams**.
//!
//! Each replicate `b` derives its own PRNG from `Xoshiro256pp::substream(seed,
//! b)`, so the result of replicate `b` depends only on `(seed, b)` — never on
//! how replicates are scheduled across threads. That is what makes the bootstrap
//! output bit-identical regardless of `RAYON_NUM_THREADS` (see the
//! determinism test). With the `parallel` feature off, the same closure runs
//! serially with identical results.

use panelkit_linalg::rng::Xoshiro256pp;

/// Run `f(b, rng_b)` for `b in 0..n_reps`, each with an independent seeded
/// substream, collecting results in replicate order.
pub fn par_map<T, F>(n_reps: usize, seed: u64, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &mut Xoshiro256pp) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n_reps)
            .into_par_iter()
            .map(|b| {
                let mut rng = Xoshiro256pp::substream(seed, b as u64);
                f(b, &mut rng)
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n_reps)
            .map(|b| {
                let mut rng = Xoshiro256pp::substream(seed, b as u64);
                f(b, &mut rng)
            })
            .collect()
    }
}

/// Parallel map over a fixed work-list (no RNG), order-preserving. Used by the
/// placebo / jackknife engines, whose per-item results are independent of order.
pub fn par_map_items<I, T, F>(items: Vec<I>, f: F) -> Vec<T>
where
    I: Send,
    T: Send,
    F: Fn(I) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.into_iter().map(f).collect()
    }
}
