//! `panelkit-estimators` — causal-inference estimators for panel data.
//!
//! Estimators are functions of a [`panel::Panel`] plus a config struct. They
//! return point estimates *and* the structural objects (weights, counterfactual
//! paths, influence functions) that downstream inference engines consume. They
//! do **not** run resampling themselves — that lives in `panelkit-inference`, so
//! any fit composes with any valid inference engine.

pub mod panel;
pub mod result;
pub mod sc;

pub use panel::Panel;
pub use result::{DidFit, ScFit};
