//! Householder QR factorization `A = Q R` and least-squares solving.
//!
//! Used for OLS / ridge outcome regressions where conditioning matters more than
//! raw speed (TWFE, Sun–Abraham, the SDID final regression). Reflectors are
//! stored compactly: column `j`'s reflector tail lives below the diagonal of the
//! packed factor, `R` in the upper triangle, with the betas held separately.

use crate::error::{LinalgError, Result};
use crate::matrix::Mat;
use crate::ops::transform::Householder;

/// A packed Householder QR factorization of an `m×n` matrix (`m >= n`).
pub struct Qr {
    /// Packed factor: upper triangle is `R`; below-diagonal of column `j` holds
    /// the tail (`v[1..]`) of reflector `j`, whose leading component is 1.
    packed: Mat,
    betas: Vec<f64>,
    m: usize,
    n: usize,
}

impl Qr {
    /// Compute the QR factorization. Requires `m >= n`.
    pub fn new(a: &Mat) -> Result<Qr> {
        let m = a.rows();
        let n = a.cols();
        if m < n {
            return Err(LinalgError::DimMismatch {
                expected: n,
                got: m,
            });
        }
        let mut packed = a.clone();
        let mut betas = vec![0.0; n];

        for j in 0..n {
            // Reflector from the column tail packed[j.., j].
            let x: Vec<f64> = packed.col(j)[j..].to_vec();
            let h = Householder::new(&x);
            betas[j] = h.beta;

            // Apply H to trailing columns j+1..n.
            for c in (j + 1)..n {
                let col = &mut packed.data[c * m..(c + 1) * m];
                // w = beta * (v · col[j..])
                let mut w = 0.0;
                for k in 0..h.v.len() {
                    w += h.v[k] * col[j + k];
                }
                w *= h.beta;
                for k in 0..h.v.len() {
                    col[j + k] -= w * h.v[k];
                }
            }

            // Store R[j,j] = alpha and the reflector tail below the diagonal.
            let vlen = h.v.len();
            let col = &mut packed.data[j * m..(j + 1) * m];
            col[j] = h.alpha;
            col[j + 1..j + vlen].copy_from_slice(&h.v[1..vlen]);
        }

        Ok(Qr {
            packed,
            betas,
            m,
            n,
        })
    }

    /// Apply `Qᵀ` to a vector `b` of length `m` in place.
    fn apply_qt(&self, b: &mut [f64]) {
        debug_assert_eq!(b.len(), self.m);
        for j in 0..self.n {
            let beta = self.betas[j];
            if beta == 0.0 {
                continue;
            }
            // v: implicit 1 at position j, tail below diagonal.
            let col = &self.packed.data[j * self.m..(j + 1) * self.m];
            // w = beta * (v · b[j..])
            let mut w = b[j];
            for i in (j + 1)..self.m {
                w += col[i] * b[i];
            }
            w *= beta;
            b[j] -= w;
            for i in (j + 1)..self.m {
                b[i] -= w * col[i];
            }
        }
    }

    /// Back-substitute `R x = rhs[0..n]`, returning `x` (length `n`).
    ///
    /// Householder QR does not rank-reveal, so a rank-deficient design can leave a
    /// (near-)zero pivot on the diagonal. Rather than emit `inf`/`NaN` (which would
    /// silently poison downstream OLS coefficients), we zero that component — a
    /// minimum-norm-style choice — using a relative pivot threshold.
    fn back_solve(&self, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut max_diag = 0.0_f64;
        for i in 0..n {
            max_diag = max_diag.max(self.packed.get(i, i).abs());
        }
        let eps = 1e-12 * max_diag.max(1.0);
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = rhs[i];
            for k in (i + 1)..n {
                s -= self.packed.get(i, k) * x[k];
            }
            let rii = self.packed.get(i, i);
            x[i] = if rii.abs() > eps { s / rii } else { 0.0 };
        }
        x
    }

    /// Least-squares solve `min_x ‖A x − b‖₂`, returning `x` (length `n`).
    pub fn solve_lstsq(&self, b: &[f64]) -> Vec<f64> {
        let mut qtb = b.to_vec();
        self.apply_qt(&mut qtb);
        self.back_solve(&qtb)
    }

    /// The upper-triangular factor `R` (`n×n`).
    pub fn r(&self) -> Mat {
        let mut r = Mat::zeros(self.n, self.n);
        for j in 0..self.n {
            for i in 0..=j {
                r.set(i, j, self.packed.get(i, j));
            }
        }
        r
    }

    /// Reconstruct the (thin) `Q` factor as an `m×n` matrix, by applying the
    /// reflectors to the columns of the identity. Mainly for testing.
    pub fn q_thin(&self) -> Mat {
        let mut q = Mat::zeros(self.m, self.n);
        for j in 0..self.n {
            q.set(j, j, 1.0);
        }
        // Apply reflectors in reverse order to e_j columns: Q = H_0 H_1 ... H_{n-1}.
        for c in 0..self.n {
            let col = &mut q.data[c * self.m..(c + 1) * self.m];
            for j in (0..self.n).rev() {
                let beta = self.betas[j];
                if beta == 0.0 {
                    continue;
                }
                let pj = &self.packed.data[j * self.m..(j + 1) * self.m];
                let mut w = col[j];
                for i in (j + 1)..self.m {
                    w += pj[i] * col[i];
                }
                w *= beta;
                col[j] -= w;
                for i in (j + 1)..self.m {
                    col[i] -= w * pj[i];
                }
            }
        }
        q
    }
}
