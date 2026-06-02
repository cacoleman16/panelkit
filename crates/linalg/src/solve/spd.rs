//! Symmetric-positive-definite solves via Cholesky.

use crate::error::Result;
use crate::factor::cholesky::Cholesky;
use crate::matrix::Mat;

/// Solve `A x = b` for SPD `A`, returning `x`.
pub fn solve_spd(a: &Mat, b: &[f64]) -> Result<Vec<f64>> {
    let chol = Cholesky::new(a)?;
    Ok(chol.solve_vec(b))
}

/// Solve `(A + ridge·I) x = b` for SPD `A`.
pub fn solve_spd_ridge(a: &Mat, ridge: f64, b: &[f64]) -> Result<Vec<f64>> {
    let chol = Cholesky::new_ridge(a, ridge)?;
    Ok(chol.solve_vec(b))
}
