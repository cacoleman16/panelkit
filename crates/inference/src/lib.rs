//! `panelkit-inference` — resampling inference engines.
//!
//! Engines are generic over estimators that implement a `Refittable`-style
//! contract (refit on a resampled/permuted panel and return a statistic), so the
//! same fit composes with any valid engine. All engines use deterministic,
//! per-replicate seeded substreams, which makes results bit-identical regardless
//! of thread count.

pub mod batch;
pub mod bootstrap;
pub mod ci;
pub mod parallel;
pub mod placebo;

pub use batch::{asc_att_many, sc_att_many, sdid_att_many};
pub use bootstrap::{
    block_bootstrap_mean, jackknife_se, multiplier_bootstrap, stationary_bootstrap_mean,
};
pub use ci::{normal_quantile, percentile_ci, ConfidenceInterval};
pub use parallel::{par_map, par_map_items};
pub use placebo::{sc_placebo, sc_placebo_method, PlaceboResult, ScMethod};
