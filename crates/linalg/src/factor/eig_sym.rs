//! Symmetric eigendecomposition `A = V Λ Vᵀ` via **cyclic Jacobi**.
//!
//! Only symmetric input is supported (none of panelkit's estimators need a
//! general nonsymmetric eigensolver). Cyclic Jacobi is short, unconditionally
//! convergent, and accurate; it reuses the same plane-rotation idea as the SVD.
//! Uses: spectral diagnostics (condition number, effective rank of the donor
//! Gram matrix) and the Gram-based SVD oracle in [`super::svd_gram`].

use crate::matrix::Mat;

/// Symmetric eigendecomposition. Eigenvalues are returned in non-increasing
/// order, with matching eigenvector columns in `vectors`.
pub struct SymEig {
    values: Vec<f64>,
    vectors: Mat,
}

impl SymEig {
    pub fn new(a: &Mat) -> SymEig {
        SymEig::new_tol(a, 1e-15, 100)
    }

    pub fn new_tol(a: &Mat, tol: f64, max_sweeps: usize) -> SymEig {
        let n = a.rows();
        // Hard assert (not debug): in release a non-square input would silently
        // decompose the leading square block — a wrong answer, not a crash.
        assert_eq!(n, a.cols(), "SymEig requires a square matrix");
        // Symmetrize defensively (read both triangles, average).
        let mut m = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                m.set(i, j, 0.5 * (a.get(i, j) + a.get(j, i)));
            }
        }
        let mut v = Mat::identity(n);

        let frob_off = |m: &Mat| -> f64 {
            let mut s = 0.0;
            for j in 0..n {
                for i in 0..j {
                    let x = m.get(i, j);
                    s += 2.0 * x * x;
                }
            }
            s.sqrt()
        };

        // Strictly relative convergence scale. (A `.max(1.0)` floor here would
        // turn the test absolute for small-normed matrices — a matrix with
        // norm 1e-16 would pass the "converged" check after zero sweeps and
        // return identity eigenvectors.) The zero matrix converges immediately
        // since its off-diagonal norm is exactly 0 ≤ 0.
        let scale = {
            let mut s = 0.0;
            for j in 0..n {
                for i in 0..n {
                    s += m.get(i, j) * m.get(i, j);
                }
            }
            s.sqrt()
        };

        for _ in 0..max_sweeps {
            if frob_off(&m) <= tol * scale {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = m.get(p, q);
                    if apq == 0.0 {
                        continue;
                    }
                    let app = m.get(p, p);
                    let aqq = m.get(q, q);
                    let theta = (aqq - app) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;

                    // Apply J' A J (two-sided rotation) updating rows/cols p,q.
                    for k in 0..n {
                        let akp = m.get(k, p);
                        let akq = m.get(k, q);
                        m.set(k, p, c * akp - s * akq);
                        m.set(k, q, s * akp + c * akq);
                    }
                    for k in 0..n {
                        let apk = m.get(p, k);
                        let aqk = m.get(q, k);
                        m.set(p, k, c * apk - s * aqk);
                        m.set(q, k, s * apk + c * aqk);
                    }
                    // Accumulate eigenvectors.
                    for k in 0..n {
                        let vkp = v.get(k, p);
                        let vkq = v.get(k, q);
                        v.set(k, p, c * vkp - s * vkq);
                        v.set(k, q, s * vkp + c * vkq);
                    }
                }
            }
        }

        let mut values: Vec<f64> = (0..n).map(|i| m.get(i, i)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        // total_cmp: NaN input must yield NaN output, not a comparator panic.
        order.sort_by(|&i, &j| values[j].total_cmp(&values[i]));

        let mut vsorted = Mat::zeros(n, n);
        let mut sorted_vals = vec![0.0; n];
        for (newj, &oldj) in order.iter().enumerate() {
            vsorted.col_mut(newj).copy_from_slice(v.col(oldj));
            sorted_vals[newj] = values[oldj];
        }
        values = sorted_vals;

        SymEig {
            values,
            vectors: vsorted,
        }
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn vectors(&self) -> &Mat {
        &self.vectors
    }
}
