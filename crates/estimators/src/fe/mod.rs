//! Fixed-effects machinery shared by the DiD estimators.

pub mod within;

pub use within::{grand_mean, time_means, two_way_within, unit_means, unit_within};
