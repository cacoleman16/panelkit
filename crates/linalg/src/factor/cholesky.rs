//! Cholesky factorization `A = L Lᵀ` for symmetric positive-definite `A`.
//!
//! This is the workhorse SPD solver: ridge regressions (ASC, SDID weights),
//! normal-equation OLS, and any Gram-matrix system route through here. Add a
//! ridge term to the diagonal (`A + λI`) to keep marginally-conditioned systems
//! SPD.

use crate::error::{LinalgError, Result};
use crate::matrix::Mat;

/// A computed Cholesky factor. `l` is lower-triangular with `A = L Lᵀ`.
pub struct Cholesky {
    l: Mat,
}

impl Cholesky {
    /// Factor a symmetric positive-definite matrix. Only the lower triangle of
    /// `a` is read. Returns [`LinalgError::NotPositiveDefinite`] at the first
    /// non-positive pivot.
    pub fn new(a: &Mat) -> Result<Cholesky> {
        let n = a.rows();
        if n != a.cols() {
            return Err(LinalgError::NotSquare {
                rows: a.rows(),
                cols: a.cols(),
            });
        }
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            // d = a[j,j] - sum_{k<j} l[j,k]^2
            let mut d = a.get(j, j);
            for k in 0..j {
                let ljk = l.get(j, k);
                d -= ljk * ljk;
            }
            if d <= 0.0 || !d.is_finite() {
                return Err(LinalgError::NotPositiveDefinite { pivot: j });
            }
            let ljj = d.sqrt();
            l.set(j, j, ljj);
            // For i > j: l[i,j] = (a[i,j] - sum_{k<j} l[i,k] l[j,k]) / l[j,j]
            for i in (j + 1)..n {
                let mut s = a.get(i, j);
                for k in 0..j {
                    s -= l.get(i, k) * l.get(j, k);
                }
                l.set(i, j, s / ljj);
            }
        }
        Ok(Cholesky { l })
    }

    /// Factor `A + ridge * I` (a convenience for regularized normal equations).
    pub fn new_ridge(a: &Mat, ridge: f64) -> Result<Cholesky> {
        let mut m = a.clone();
        for i in 0..m.rows() {
            m.add_to(i, i, ridge);
        }
        Cholesky::new(&m)
    }

    /// The lower-triangular factor `L`.
    pub fn l(&self) -> &Mat {
        &self.l
    }

    /// Solve `A x = b` in place-free form, returning `x`. `b` length must equal `n`.
    pub fn solve_vec(&self, b: &[f64]) -> Vec<f64> {
        let n = self.l.rows();
        debug_assert_eq!(b.len(), n);
        let mut y = b.to_vec();
        // Forward solve L y = b.
        for i in 0..n {
            let mut s = y[i];
            for k in 0..i {
                s -= self.l.get(i, k) * y[k];
            }
            y[i] = s / self.l.get(i, i);
        }
        // Back solve Lᵀ x = y.
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= self.l.get(k, i) * y[k];
            }
            y[i] = s / self.l.get(i, i);
        }
        y
    }

    /// Solve `A X = B` for a matrix right-hand side, column by column.
    pub fn solve_mat(&self, b: &Mat) -> Mat {
        let n = self.l.rows();
        debug_assert_eq!(b.rows(), n);
        let mut x = Mat::zeros(n, b.cols());
        for j in 0..b.cols() {
            let xj = self.solve_vec(b.col(j));
            x.col_mut(j).copy_from_slice(&xj);
        }
        x
    }

    /// Log-determinant of `A` (= `2 Σ log L_ii`). Useful for likelihoods.
    pub fn logdet(&self) -> f64 {
        let n = self.l.rows();
        let mut s = 0.0;
        for i in 0..n {
            s += self.l.get(i, i).ln();
        }
        2.0 * s
    }
}
