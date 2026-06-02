//! The synthetic-control family: classic SC, augmented SC, and synthetic DiD.

pub mod synthetic;

pub use synthetic::{fit as fit_sc, fit_at, fit_series, ScConfig};
