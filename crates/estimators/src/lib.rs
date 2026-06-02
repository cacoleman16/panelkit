//! `panelkit-estimators` — causal-inference estimators for panel data.
//!
//! Estimators are functions of a [`panel::Panel`] plus a config struct. They
//! return point estimates *and* the structural objects (weights, counterfactual
//! paths, influence functions) that downstream inference engines consume. They
//! do **not** run resampling themselves — that lives in `panelkit-inference`, so
//! any fit composes with any valid inference engine.

// Index-based loops over panel dimensions read more clearly than zipped
// iterators in this numeric code; opt out of the lint crate-wide.
#![allow(clippy::needless_range_loop)]

pub mod did;
pub mod fe;
pub mod mcnnm;
pub mod panel;
pub mod result;
pub mod sc;

pub use panel::Panel;
pub use result::{DidFit, ScFit};
