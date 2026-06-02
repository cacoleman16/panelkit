//! Singular value decomposition `A = U Σ Vᵀ` via **one-sided Jacobi**
//! (Hestenes).
//!
//! Why one-sided Jacobi rather than Golub–Kahan: it is dramatically easier to
//! implement correctly, and it computes small singular values to high *relative*
//! accuracy (Demmel–Veselić) — exactly what the MC-NNM soft-thresholding path
//! needs, where the spectrum near the threshold determines the answer. At
//! panelkit's sizes the extra flops are irrelevant. A Golub–Kahan path is kept
//! in [`crate::factor::bidiag`] purely as an independent test oracle.
//!
//! The method orthogonalizes the columns of (a copy of) `A` by a sequence of
//! plane rotations chosen to annihilate column-pair inner products; the column
//! norms at convergence are the singular values, the normalized columns are `U`,
//! and the accumulated rotations form `V`.

use crate::matrix::Mat;
use crate::ops::norms::nrm2;

/// A thin SVD: `U` is `m×k`, `s` has length `k`, `V` is `n×k`, with
/// `k = min(m, n)` and singular values in non-increasing order.
pub struct Svd {
    u: Mat,
    s: Vec<f64>,
    v: Mat,
    m: usize,
    n: usize,
}

impl Svd {
    /// Compute the thin SVD. Internally works on the tall orientation
    /// (`rows >= cols`) and swaps roles if `A` is wide.
    pub fn new(a: &Mat) -> Svd {
        Svd::new_tol(a, 1e-15, 60)
    }

    /// Compute with an explicit off-diagonal tolerance and sweep cap.
    pub fn new_tol(a: &Mat, tol: f64, max_sweeps: usize) -> Svd {
        let (m0, n0) = a.shape();
        let transposed = m0 < n0;
        // Work on the tall version W (rows >= cols).
        let w = if transposed { a.transpose() } else { a.clone() };
        let m = w.rows();
        let n = w.cols();

        let mut u = w; // columns will become U * Σ
        let mut v = Mat::identity(n);

        // One-sided Jacobi sweeps.
        for _sweep in 0..max_sweeps {
            let mut converged = true;
            for p in 0..n {
                for q in (p + 1)..n {
                    let (alpha, beta, gamma) = {
                        let up = &u.data[p * m..(p + 1) * m];
                        let uq = &u.data[q * m..(q + 1) * m];
                        let mut a_pp = 0.0;
                        let mut a_qq = 0.0;
                        let mut a_pq = 0.0;
                        for i in 0..m {
                            a_pp += up[i] * up[i];
                            a_qq += uq[i] * uq[i];
                            a_pq += up[i] * uq[i];
                        }
                        (a_pp, a_qq, a_pq)
                    };

                    // Skip if columns already (numerically) orthogonal.
                    if gamma.abs() <= tol * (alpha * beta).sqrt() {
                        continue;
                    }
                    converged = false;

                    // Jacobi rotation that diagonalizes [[alpha, gamma],[gamma, beta]].
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = c * t;

                    // Rotate columns p, q of U and V.
                    rotate_cols(&mut u.data, m, p, q, c, s);
                    rotate_cols(&mut v.data, n, p, q, c, s);
                }
            }
            if converged {
                break;
            }
        }

        // Column norms are the singular values; normalize to get U.
        let mut s = vec![0.0; n];
        for j in 0..n {
            let col = &u.data[j * m..(j + 1) * m];
            let nrm = nrm2(col);
            s[j] = nrm;
            if nrm > 0.0 {
                let inv = 1.0 / nrm;
                let col = &mut u.data[j * m..(j + 1) * m];
                for x in col.iter_mut() {
                    *x *= inv;
                }
            }
        }

        // Sort singular values (and U, V columns) in non-increasing order.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());

        let mut u_sorted = Mat::zeros(m, n);
        let mut v_sorted = Mat::zeros(n, n);
        let mut s_sorted = vec![0.0; n];
        for (newj, &oldj) in order.iter().enumerate() {
            u_sorted
                .col_mut(newj)
                .copy_from_slice(&u.data[oldj * m..(oldj + 1) * m]);
            v_sorted
                .col_mut(newj)
                .copy_from_slice(&v.data[oldj * n..(oldj + 1) * n]);
            s_sorted[newj] = s[oldj];
        }

        // Thin to k = min(m, n) columns.
        let k = m.min(n);
        let u_thin = u_sorted.cols_range(0, k);
        let v_thin = v_sorted.cols_range(0, k);
        s_sorted.truncate(k);

        if transposed {
            // A = Aᵀᵀ; swap roles of U and V.
            Svd {
                u: v_thin,
                s: s_sorted,
                v: u_thin,
                m: m0,
                n: n0,
            }
        } else {
            Svd {
                u: u_thin,
                s: s_sorted,
                v: v_thin,
                m: m0,
                n: n0,
            }
        }
    }

    /// Singular values, non-increasing, length `k = min(m, n)`.
    pub fn singular_values(&self) -> &[f64] {
        &self.s
    }

    /// Left singular vectors `U` (`m×k`).
    pub fn u(&self) -> &Mat {
        &self.u
    }

    /// Right singular vectors `V` (`n×k`).
    pub fn v(&self) -> &Mat {
        &self.v
    }

    /// Numerical rank: count of singular values above `tol * σ_max`.
    pub fn rank(&self, rel_tol: f64) -> usize {
        let smax = self.s.first().copied().unwrap_or(0.0);
        let thresh = rel_tol * smax;
        self.s.iter().filter(|&&x| x > thresh).count()
    }

    /// Reconstruct `U diag(d) Vᵀ` for an arbitrary diagonal `d` (length `k`).
    /// Used by the SVT operator (pass soft-thresholded singular values).
    pub fn reconstruct_with(&self, d: &[f64]) -> Mat {
        let k = self.s.len();
        debug_assert_eq!(d.len(), k);
        let mut out = Mat::zeros(self.m, self.n);
        // out += Σ d_t u_t v_tᵀ
        for t in 0..k {
            let dt = d[t];
            if dt == 0.0 {
                continue;
            }
            let ut = &self.u.data[t * self.m..(t + 1) * self.m];
            let vt = &self.v.data[t * self.n..(t + 1) * self.n];
            for jj in 0..self.n {
                let vtj = dt * vt[jj];
                if vtj == 0.0 {
                    continue;
                }
                let ocol = &mut out.data[jj * self.m..(jj + 1) * self.m];
                for ii in 0..self.m {
                    ocol[ii] += vtj * ut[ii];
                }
            }
        }
        out
    }

    /// Reconstruct the original matrix `A = U Σ Vᵀ` (for testing).
    pub fn reconstruct(&self) -> Mat {
        self.reconstruct_with(&self.s)
    }
}

/// In-place rotation of columns `p` and `q` of a column-major buffer with
/// leading dimension `lead`: `[col_p, col_q] · [[c, s], [-s, c]]`.
#[inline]
fn rotate_cols(data: &mut [f64], lead: usize, p: usize, q: usize, c: f64, s: f64) {
    debug_assert!(p < q);
    // Split the borrow so we can touch both columns at once.
    let (left, right) = data.split_at_mut(q * lead);
    let cp = &mut left[p * lead..(p + 1) * lead];
    let cq = &mut right[..lead];
    for i in 0..lead {
        let xp = cp[i];
        let xq = cq[i];
        cp[i] = c * xp - s * xq;
        cq[i] = s * xp + c * xq;
    }
}
