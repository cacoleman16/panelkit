//! The difference-in-differences family: two-way fixed effects, Callaway &
//! Sant'Anna group-time ATTs, and the Sun & Abraham interaction-weighted event
//! study.

pub mod bacon;
pub mod callaway;
pub mod sunab;
pub mod twfe;

pub use bacon::{decompose as bacon_decompose, BaconComponent, BaconKind, BaconResult};
pub use callaway::{fit as fit_callaway, AggEffect, CsResult, GroupTimeAtt};
pub use sunab::{fit as fit_sunab, SaResult};
pub use twfe::{fit as fit_twfe, treatment_matrix, TwfeFit};
