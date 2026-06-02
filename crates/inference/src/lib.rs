//! `panelkit-inference` — resampling inference engines.
//!
//! Engines are generic over estimators that implement a `Refittable`-style
//! contract (refit on a resampled/permuted panel and return a statistic), so the
//! same fit composes with any valid engine. All engines use deterministic,
//! per-replicate seeded substreams, which makes results bit-identical regardless
//! of thread count.

pub mod ci;
pub mod placebo;

pub use ci::{percentile_ci, ConfidenceInterval};
pub use placebo::{sc_placebo, PlaceboResult};
