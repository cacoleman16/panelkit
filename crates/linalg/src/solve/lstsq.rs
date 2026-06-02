//! Ordinary and ridge least squares.
//!
//! [`ols`] uses QR (stable, the default for design matrices that may be
//! ill-conditioned). [`ols_normal`] and [`ridge`] use the Cholesky-on-Gram
//! route, which is faster and is the natural form for ridge (`XᵀX + λI` is SPD
//! by construction).

use crate::error::Result;
use crate::factor::cholesky::Cholesky;
use crate::factor::qr::Qr;
use crate::matrix::Mat;
use crate::ops::matmul::{matvec, matvec_t, syrk_ata};
use crate::ops::norms::nrm2;

/// Solve `min_b ‖X b − y‖₂` via QR. `X` is `n×p` with `n >= p`.
pub fn ols(x: &Mat, y: &[f64]) -> Result<Vec<f64>> {
    let qr = Qr::new(x)?;
    Ok(qr.solve_lstsq(y))
}

/// OLS via the normal equations `XᵀX b = Xᵀy` (Cholesky). Faster than QR but
/// squares the condition number — prefer [`ols`] when conditioning is in doubt.
pub fn ols_normal(x: &Mat, y: &[f64]) -> Result<Vec<f64>> {
    let xtx = syrk_ata(x);
    let xty = matvec_t(x, y);
    let chol = Cholesky::new(&xtx)?;
    Ok(chol.solve_vec(&xty))
}

/// Result of a ridge fit, carrying the coefficients and enough to reproduce
/// fitted values / residuals.
pub struct RidgeFit {
    pub coef: Vec<f64>,
}

impl RidgeFit {
    /// Fitted values `X b`.
    pub fn fitted(&self, x: &Mat) -> Vec<f64> {
        matvec(x, &self.coef)
    }

    /// Residuals `y − X b`.
    pub fn residuals(&self, x: &Mat, y: &[f64]) -> Vec<f64> {
        let f = self.fitted(x);
        y.iter().zip(f.iter()).map(|(a, b)| a - b).collect()
    }

    /// Residual sum of squares.
    pub fn rss(&self, x: &Mat, y: &[f64]) -> f64 {
        let r = self.residuals(x, y);
        nrm2(&r).powi(2)
    }
}

/// Ridge regression: `b = (XᵀX + λI)⁻¹ Xᵀy`, solved (not inverted) via Cholesky.
pub fn ridge(x: &Mat, y: &[f64], lambda: f64) -> Result<RidgeFit> {
    let xtx = syrk_ata(x);
    let xty = matvec_t(x, y);
    let chol = Cholesky::new_ridge(&xtx, lambda)?;
    Ok(RidgeFit {
        coef: chol.solve_vec(&xty),
    })
}
