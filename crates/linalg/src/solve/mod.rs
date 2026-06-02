//! Linear solves built on the factorizations. Rule of thumb: SPD systems and
//! regularized normal equations go through Cholesky; potentially
//! ill-conditioned least squares go through QR. We never form an explicit
//! inverse — always solve.

pub mod lstsq;
pub mod spd;

pub use lstsq::{ols, ols_normal, ridge, RidgeFit};
pub use spd::solve_spd;
