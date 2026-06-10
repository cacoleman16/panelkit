//! Solvers for the simplex-constrained quadratic program at the heart of
//! synthetic-control weight estimation:
//!
//! ```text
//!   minimize   ½ wᵀ(G + η I) w − bᵀ w      subject to   w ≥ 0,  Σ w = 1
//! ```
//!
//! For the synthetic-control fit, `G = Y₀ᵀ Y₀`, `b = Y₀ᵀ y`, so the objective is
//! `½‖y − Y₀ w‖²` (plus an optional ridge `η`).
//!
//! Two independent methods are provided and are expected to agree on the
//! optimum (a cross-check exploited in tests):
//! - [`solve_fw`]: Frank–Wolfe / conditional gradient — the simplex
//!   linear-minimization oracle is a trivial `argmin` over the gradient, no
//!   projection needed, and it naturally yields sparse vertex solutions.
//! - [`solve_pg`]: projected gradient with the Duchi/Condat simplex projection —
//!   the natural base for the SDID weight problems that carry extra structure.

use crate::matrix::Mat;
use crate::ops::matmul::matvec;
use crate::ops::norms::dot;

/// Euclidean projection of `v` onto the probability simplex `{w ≥ 0, Σw = 1}`
/// (Duchi et al. 2008 sort-based algorithm).
pub fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a)); // descending; NaN-safe
    let mut css = 0.0;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

/// Gradient `g = (G + ηI) w − b` of the simplex QP.
fn grad(gram: &Mat, b: &[f64], eta: f64, w: &[f64]) -> Vec<f64> {
    let mut g = matvec(gram, w);
    for i in 0..g.len() {
        g[i] += eta * w[i] - b[i];
    }
    g
}

/// Result of a simplex QP solve.
pub struct SimplexSolution {
    pub w: Vec<f64>,
    pub iters: usize,
    /// Final Frank–Wolfe duality gap (≈ 0 at the optimum).
    pub gap: f64,
}

/// **Away-step** Frank–Wolfe solver for the simplex QP. `gram` is `J×J`
/// SPD(-ish), `b` length `J`, `eta` an optional ridge on the weights.
///
/// Plain Frank–Wolfe zig-zags and converges only sublinearly when the optimum
/// lies on a low-dimensional face — precisely the sparse-weight regime synthetic
/// control lands in. The away-step variant adds a second candidate direction
/// (moving *away* from the worst-aligned active vertex), recovering linear
/// convergence and reaching faces exactly. Because the simplex vertices are the
/// unit basis vectors, the iterate `w` *is* its own barycentric-weight vector,
/// so the active set is simply `{i : w_i > 0}` and no extra bookkeeping is
/// needed.
pub fn solve_fw(gram: &Mat, b: &[f64], eta: f64, max_iter: usize, tol: f64) -> SimplexSolution {
    let j = b.len();
    debug_assert_eq!(gram.rows(), j);
    if j == 0 {
        return SimplexSolution {
            w: Vec::new(),
            iters: 0,
            gap: 0.0,
        };
    }
    // Start at the simplex barycenter (all vertices active).
    let mut w = vec![1.0 / j as f64; j];
    let mut last_gap = f64::INFINITY;
    let drop_tol = 1e-14;

    for k in 0..max_iter {
        let g = grad(gram, b, eta, &w);

        // Frank–Wolfe vertex: s = argmin_i g_i over all vertices.
        let mut s = 0usize;
        let mut gmin = g[0];
        for i in 1..j {
            if g[i] < gmin {
                gmin = g[i];
                s = i;
            }
        }
        // Away vertex: v = argmax_i g_i over the *active* set (w_i > 0).
        let mut v = usize::MAX;
        let mut gmax = f64::NEG_INFINITY;
        for i in 0..j {
            if w[i] > drop_tol && g[i] > gmax {
                gmax = g[i];
                v = i;
            }
        }

        // FW duality gap = −gᵀd_FW = g·w − g_s. This is the true stopping crit.
        let gw = dot(&g, &w);
        let gap = gw - gmin;
        last_gap = gap;
        if gap <= tol {
            return SimplexSolution { w, iters: k, gap };
        }

        // FW direction d_FW = e_s − w; away direction d_A = w − e_v.
        // Descent ∝ −g·d. Pick whichever direction descends more.
        let fw_descent = -(gmin - gw); // = gw − gmin = gap  (>= 0)
        let away_descent = gmax - gw; // = −g·d_A
        let (d, gamma_max) = if v == usize::MAX || fw_descent >= away_descent {
            // Frank–Wolfe step, step size in [0, 1].
            let mut d = w.iter().map(|&wi| -wi).collect::<Vec<_>>();
            d[s] += 1.0;
            (d, 1.0)
        } else {
            // Away step, step size in [0, w_v / (1 − w_v)].
            let mut d = w.clone();
            d[v] -= 1.0;
            let gmax_step = if w[v] < 1.0 {
                w[v] / (1.0 - w[v])
            } else {
                // Degenerate: the away vertex carries all mass; cap the step.
                f64::INFINITY
            };
            (d, gmax_step)
        };

        // Exact line search: γ* = −(g·d) / (dᵀ(G+ηI)d), clamped to [0, γ_max].
        let gd = dot(&g, &d);
        let mut gd_vec = matvec(gram, &d);
        for i in 0..j {
            gd_vec[i] += eta * d[i];
        }
        let dgd = dot(&d, &gd_vec);
        let gamma = if dgd > 0.0 {
            (-gd / dgd).clamp(0.0, gamma_max)
        } else if gd < 0.0 {
            // Non-positive curvature along a descent direction → go to the
            // feasible cap (bounded so the step never leaves the simplex).
            gamma_max.min(1.0)
        } else {
            // Not a descent direction → don't move.
            0.0
        };
        for i in 0..j {
            w[i] += gamma * d[i];
        }
        // Clean up tiny negatives / renormalize against accumulated round-off.
        let mut sum = 0.0;
        for wi in w.iter_mut() {
            if *wi < 0.0 {
                *wi = 0.0;
            }
            sum += *wi;
        }
        if sum > 0.0 && (sum - 1.0).abs() > 1e-15 {
            let inv = 1.0 / sum;
            for wi in w.iter_mut() {
                *wi *= inv;
            }
        }
    }

    SimplexSolution {
        w,
        iters: max_iter,
        gap: last_gap,
    }
}

/// Projected-gradient solver for the simplex QP, with a fixed step `1/L` where
/// `L` is a guaranteed upper bound on the curvature of `G + ηI` (Gershgorin).
/// Used where the FW vertex bias is undesirable.
pub fn solve_pg(gram: &Mat, b: &[f64], eta: f64, max_iter: usize, tol: f64) -> SimplexSolution {
    let j = b.len();
    if j == 0 {
        return SimplexSolution {
            w: Vec::new(),
            iters: 0,
            gap: 0.0,
        };
    }
    // Lipschitz bound L = max_i Σ_j |G_ij| + η ≥ λ_max(G + ηI) (Gershgorin).
    // This must be an UPPER bound: power iteration converges from *below*, and
    // from a deterministic start it can converge to a smaller eigenvalue
    // entirely (e.g. when the top eigenvector is orthogonal to the all-ones
    // start), making the step too long and the iteration oscillate around a
    // wrong point.
    let l = gershgorin_max_eig(gram, eta).max(1e-12);
    let step = 1.0 / l;

    let mut w = vec![1.0 / j as f64; j];
    let mut last = f64::INFINITY;
    for k in 0..max_iter {
        let g = grad(gram, b, eta, &w);
        let trial: Vec<f64> = w
            .iter()
            .zip(g.iter())
            .map(|(wi, gi)| wi - step * gi)
            .collect();
        let wnext = project_simplex(&trial);
        // Convergence: movement size.
        let mut mv = 0.0;
        for i in 0..j {
            let d = wnext[i] - w[i];
            mv += d * d;
        }
        w = wnext;
        last = mv.sqrt();
        if last <= tol {
            return SimplexSolution {
                w,
                iters: k,
                gap: last,
            };
        }
    }
    SimplexSolution {
        w,
        iters: max_iter,
        gap: last,
    }
}

/// Gershgorin upper bound on the largest eigenvalue of `G + ηI`:
/// `max_i Σ_j |G_ij| + η`. For a symmetric PSD Gram this overshoots λ_max by at
/// most a factor of ~n — a safe (if conservative) projected-gradient step.
fn gershgorin_max_eig(gram: &Mat, eta: f64) -> f64 {
    let n = gram.rows();
    let mut l = 0.0_f64;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            row += gram.get(i, j).abs();
        }
        l = l.max(row);
    }
    l + eta
}

/// Convenience: synthetic-control weights minimizing `‖y − Y₀ w‖²` over the
/// simplex, with optional ridge `eta`. `y0` is `m×J` (pre-period donors),
/// `y` length `m` (pre-period treated outcome).
pub fn sc_weights(y0: &Mat, y: &[f64], eta: f64) -> SimplexSolution {
    let gram = crate::ops::matmul::syrk_ata(y0);
    let b = crate::ops::matmul::matvec_t(y0, y);
    solve_fw(&gram, &b, eta, 5000, 1e-10)
}
