//! `panelkit-inference` — resampling inference engines.
//!
//! Engines are generic over estimators that implement a `Refittable`-style
//! contract (refit on a resampled/permuted panel and return a statistic), so the
//! same fit composes with any valid engine. All engines use deterministic,
//! per-replicate seeded substreams, which makes results bit-identical regardless
//! of thread count.

pub mod bootstrap;
pub mod ci;
pub mod parallel;
pub mod placebo;

pub use bootstrap::{
    block_bootstrap_mean, jackknife_se, multiplier_bootstrap, stationary_bootstrap_mean,
};
pub use ci::{percentile_ci, ConfidenceInterval};
pub use parallel::{par_map, par_map_items};
pub use placebo::{sc_placebo, PlaceboResult};
