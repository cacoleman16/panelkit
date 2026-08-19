//! The synthetic-control family: classic SC, augmented SC, synthetic DiD, the
//! Ferman-Pinto demeaned estimator and spectral (robust) SC.

pub mod augmented;
pub mod cpasc;
pub mod demeaned;
pub mod robust;
pub mod sdid;
pub mod synthetic;

pub use augmented::{
    fit as fit_asc, fit_at as fit_asc_at, fit_series as asc_fit_series, AscConfig,
};
pub use cpasc::{
    fit as fit_cpasc, fit_at as fit_cpasc_at, CpascConfig, CpascFit, PoolMode, UnitFit,
};
pub use demeaned::{fit as fit_fp, fit_at as fit_fp_at, fit_series as fp_fit_series, FpConfig};
pub use robust::{fit as fit_rsc, fit_at as fit_rsc_at, fit_series as rsc_fit_series, RscConfig};
pub use sdid::{
    fit as fit_sdid, fit_at as fit_sdid_at, jackknife_loo_atts as sdid_jackknife_loo_atts,
    SdidConfig,
};
pub use synthetic::{fit as fit_sc, fit_at, fit_series, fit_series_cfg, solve_weights, ScConfig};
