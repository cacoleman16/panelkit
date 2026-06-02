//! The synthetic-control family: classic SC, augmented SC, and synthetic DiD.

pub mod augmented;
pub mod cpasc;
pub mod sdid;
pub mod synthetic;

pub use augmented::{
    fit as fit_asc, fit_at as fit_asc_at, fit_series as asc_fit_series, AscConfig,
};
pub use cpasc::{
    fit as fit_cpasc, fit_at as fit_cpasc_at, CpascConfig, CpascFit, PoolMode, UnitFit,
};
pub use sdid::{fit as fit_sdid, fit_at as fit_sdid_at, SdidConfig};
pub use synthetic::{fit as fit_sc, fit_at, fit_series, ScConfig};
