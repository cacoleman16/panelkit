//! Matrix completion estimators.

pub mod softimpute;

pub use softimpute::{fit as fit_mcnnm, fit_at as fit_mcnnm_at, McnnmConfig};
