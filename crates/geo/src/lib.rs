//! `panelkit-geo` — geo-experiment **design**: power analysis, market selection,
//! and real-world diagnostics, built on panelkit's estimators.
//!
//! This is the planning layer that sits in front of a geo test: *which* markets
//! should I treat, *how big* a lift can I detect, and *can I trust* this design?
//! It powers a test the realistic way — historical placebo with injected lift on
//! the real panel — across the SC, ASC, and SDID estimators, and surfaces
//! holdout, fit quality, improvement over a naive benchmark, seasonality, and
//! stability warnings.
//!
//! All the heavy simulation runs in Rust (parallel via the `parallel` feature);
//! report rendering and plotting live in the Python layer.

// Index-based loops over panel dimensions read more clearly than zipped
// iterators in this numeric code; opt out of the lint crate-wide.
#![allow(clippy::needless_range_loop)]

pub mod diagnostics;
pub mod power;
pub mod selection;
pub mod types;

pub use diagnostics::diagnostics;
pub use power::power_curve;
pub use selection::{evaluate, select_markets, MarketCandidate, SelectConfig};
pub use types::{Diagnostics, Method, PowerPoint, PowerResult};
