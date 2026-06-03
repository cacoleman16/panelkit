//! Randomized truncated SVD (Halko–Martinsson–Tropp).
//!
//! For the MC-NNM / SoftImpute inner loop we only need the *top* singular
//! components (everything below the threshold λ is zeroed anyway), so computing
//! a full SVD each iteration is wasteful. The randomized SVD finds a rank-`k`
//! approximation cheaply, and — crucially — stays **dependency-free**: it reuses
//! panelkit's own Householder QR and one-sided Jacobi SVD (on a small `l×n`
//! matrix), so it is correct by construction rather than linking LAPACK.
//!
//! Sketch: draw a random `n×l` test matrix `Ω` (`l = k + oversample`), form the
//! sample `Y = (A Aᵀ)^q A Ω`, orthonormalize `Q = qr(Y)`, project
//! `B = Qᵀ A` (small), SVD `B = Ũ Σ Vᵀ`, and lift `U = Q Ũ`.

use crate::factor::qr::Qr;
use crate::factor::svd::Svd;
use crate::matrix::Mat;
use crate::ops::matmul::{matmul, matvec_t};
use crate::rng::Xoshiro256pp;

/// A rank-`k` randomized SVD: `U` is `m×k`, `s` length `k`, `V` is `n×k`.
pub struct RandomizedSvd {
    pub u: Mat,
    pub s: Vec<f64>,
    pub v: Mat,
}

/// Compute a randomized rank-`rank` SVD of `a`.
///
/// `oversample` (typically 5–10) improves accuracy; `n_iter` power iterations
/// (typically 1–2) sharpen the spectrum when singular values decay slowly.
/// `seed` makes it deterministic.
pub fn randomized_svd(
    a: &Mat,
    rank: usize,
    oversample: usize,
    n_iter: usize,
    seed: u64,
) -> RandomizedSvd {
    let (m, n) = a.shape();
    let k = rank.max(1).min(m.min(n));
    let l = (k + oversample).min(m.min(n));

    // Random Gaussian test matrix Ω (n×l).
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    let mut omega = Mat::zeros(n, l);
    for v in omega.as_mut_slice().iter_mut() {
        *v = rng.next_normal();
    }

    // Sample Y = A Ω  (m×l), with optional power iterations Y ← A (Aᵀ Y).
    let mut y = matmul(a, &omega);
    for _ in 0..n_iter {
        let at_y = matmul(&a.transpose(), &y); // n×l
        y = matmul(a, &at_y); // m×l
    }

    // Orthonormal basis Q (m×l) for the range of Y.
    let qr = Qr::new(&y).expect("randomized_svd: QR of sample matrix");
    let q = qr.q_thin();

    // Small projected matrix B = Qᵀ A  (l×n), then its exact SVD.
    let b = {
        // Bᵀ = Aᵀ Q  (n×l); columns are Aᵀ q_j = matvec_t(a, q_j).
        let mut bt = Mat::zeros(n, l);
        for j in 0..l {
            let col = matvec_t(a, q.col(j)); // length n
            bt.col_mut(j).copy_from_slice(&col);
        }
        bt.transpose() // l×n
    };
    let svd_b = Svd::new(&b);

    // Lift: U = Q Ũ  (m×l), then truncate to k.
    let u_full = matmul(&q, svd_b.u());
    let s_full = svd_b.singular_values();
    let v_full = svd_b.v();

    let u = u_full.cols_range(0, k);
    let v = v_full.cols_range(0, k);
    let s = s_full[..k].to_vec();
    RandomizedSvd { u, s, v }
}

impl RandomizedSvd {
    /// Reconstruct `U diag(d) Vᵀ` for an arbitrary diagonal `d` (length `k`),
    /// used by the truncated SVT operator.
    pub fn reconstruct_with(&self, d: &[f64]) -> Mat {
        let (m, _) = self.u.shape();
        let n = self.v.rows();
        let k = self.s.len();
        let mut out = Mat::zeros(m, n);
        for t in 0..k {
            let dt = d[t];
            if dt == 0.0 {
                continue;
            }
            let ut = self.u.col(t);
            let vt = self.v.col(t);
            for jj in 0..n {
                let vtj = dt * vt[jj];
                if vtj == 0.0 {
                    continue;
                }
                let ocol = &mut out.as_mut_slice()[jj * m..(jj + 1) * m];
                for ii in 0..m {
                    ocol[ii] += vtj * ut[ii];
                }
            }
        }
        out
    }
}
