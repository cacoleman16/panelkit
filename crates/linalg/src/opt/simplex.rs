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

use crate::factor::cholesky::Cholesky;
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

// ---------------------------------------------------------------------------
// Bounded ("capped") simplex: lo ≤ w_i ≤ hi, Σ w = 1.
// ---------------------------------------------------------------------------

/// Per-donor bounds on synthetic-control weights: `lo ≤ w_i ≤ hi` on top of the
/// simplex constraint `Σ w = 1`.
///
/// `hi` caps concentration (no single donor may carry more than `hi` of the
/// synthetic unit); `lo` forces diversification (every donor carries at least
/// `lo`). The default `{lo: 0, hi: 1}` is the plain simplex — no extra
/// constraint at all.
///
/// Feasibility needs `J·lo ≤ 1 ≤ J·hi` for `J` donors; [`WeightBounds::feasible_for`]
/// relaxes a bound pair that violates it (callers that can report an error to a
/// user should do so *before* calling, rather than relying on the relaxation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightBounds {
    /// Lower bound on every weight (0 = none).
    pub lo: f64,
    /// Upper bound on every weight (1 = none).
    pub hi: f64,
}

impl Default for WeightBounds {
    fn default() -> Self {
        WeightBounds { lo: 0.0, hi: 1.0 }
    }
}

impl WeightBounds {
    /// Bounds pair, with non-finite entries treated as "no bound".
    pub fn new(lo: f64, hi: f64) -> WeightBounds {
        WeightBounds {
            lo: if lo.is_finite() { lo } else { 0.0 },
            hi: if hi.is_finite() { hi } else { 1.0 },
        }
    }

    /// A cap only: `w_i ≤ hi`.
    pub fn max_weight(hi: f64) -> WeightBounds {
        WeightBounds::new(0.0, hi)
    }

    /// Whether these bounds bind at all for `j` variables (i.e. whether the
    /// feasible set is strictly smaller than the plain simplex).
    pub fn binds(&self, j: usize) -> bool {
        j > 0 && (self.lo > 0.0 || self.hi < 1.0)
    }

    /// The bounds actually enforced for `j` variables: clamped into
    /// `0 ≤ lo ≤ 1/j ≤ hi ≤ 1` so the feasible set is always non-empty.
    pub fn feasible_for(&self, j: usize) -> WeightBounds {
        if j == 0 {
            return WeightBounds::default();
        }
        let even = 1.0 / j as f64;
        WeightBounds {
            lo: self.lo.clamp(0.0, even),
            hi: self.hi.clamp(even, 1.0),
        }
    }
}

/// Euclidean projection onto `{lo ≤ w ≤ hi, Σ w = 1}`.
///
/// The solution has the water-filling form `w_i = clamp(v_i − θ, lo, hi)` for the
/// unique `θ` with `Σ w_i(θ) = 1` (`Σ w_i(θ)` is continuous and non-increasing in
/// `θ`). We bracket `θ` between the values that peg every coordinate to `hi` and
/// to `lo` respectively and bisect; a final correction spreads the round-off
/// residual over the strictly-interior coordinates so the output sums to 1 to
/// machine precision.
pub fn project_bounded_simplex(v: &[f64], bounds: WeightBounds) -> Vec<f64> {
    let j = v.len();
    if j == 0 {
        return Vec::new();
    }
    let WeightBounds { lo, hi } = bounds.feasible_for(j);
    // Bracket: θ_lo pegs everything to `hi` (sum ≥ 1), θ_hi pegs everything to
    // `lo` (sum ≤ 1). NaNs are treated as 0 so a degenerate input cannot make
    // the bracket non-finite.
    let clean: Vec<f64> = v
        .iter()
        .map(|&x| if x.is_finite() { x } else { 0.0 })
        .collect();
    let vmin = clean.iter().cloned().fold(f64::INFINITY, f64::min);
    let vmax = clean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut a = vmin - hi;
    let mut b = vmax - lo;
    let sum_at = |theta: f64| -> f64 {
        clean
            .iter()
            .map(|&x| (x - theta).clamp(lo, hi))
            .sum::<f64>()
    };
    // 100 bisections shrink the bracket by 2^-100 — far below f64 resolution.
    for _ in 0..100 {
        let mid = 0.5 * (a + b);
        if !(mid > a && mid < b) {
            break; // bracket collapsed to adjacent floats
        }
        if sum_at(mid) > 1.0 {
            a = mid;
        } else {
            b = mid;
        }
    }
    let theta = 0.5 * (a + b);
    let mut w: Vec<f64> = clean.iter().map(|&x| (x - theta).clamp(lo, hi)).collect();
    // Spread the residual over strictly-interior coordinates (moving a pegged
    // one would leave the box).
    let resid = 1.0 - w.iter().sum::<f64>();
    if resid != 0.0 {
        let free: Vec<usize> = (0..j).filter(|&i| w[i] > lo && w[i] < hi).collect();
        if !free.is_empty() {
            let share = resid / free.len() as f64;
            for &i in &free {
                w[i] = (w[i] + share).clamp(lo, hi);
            }
        }
    }
    w
}

/// Linear-minimization oracle over `{lo ≤ s ≤ hi, Σ s = 1}`: the vertex
/// minimizing `gᵀs`. Fill every coordinate to `lo`, then hand the remaining mass
/// `1 − J·lo` to the coordinates with the smallest gradient, `hi − lo` at a time.
fn lmo_bounded(g: &[f64], bounds: WeightBounds) -> Vec<f64> {
    let j = g.len();
    let WeightBounds { lo, hi } = bounds.feasible_for(j);
    let mut s = vec![lo; j];
    let mut order: Vec<usize> = (0..j).collect();
    order.sort_by(|&x, &y| g[x].total_cmp(&g[y]));
    let mut rest = 1.0 - lo * j as f64;
    let room = hi - lo;
    for &i in &order {
        if rest <= 0.0 {
            break;
        }
        let add = room.min(rest);
        s[i] += add;
        rest -= add;
    }
    s
}

/// Upper bound on `λ_max(G + ηI)`: the tighter of the Gershgorin row-sum bound
/// and the Frobenius norm (`λ_max ≤ ‖G‖_F` for symmetric `G`). Both are strict
/// upper bounds, so the smaller is still safe as a gradient-step Lipschitz
/// constant — and for a Gram with a decaying spectrum, Frobenius is far tighter.
fn lipschitz_bound(gram: &Mat, eta: f64) -> f64 {
    let n = gram.rows();
    let mut gersh = 0.0_f64;
    let mut frob = 0.0_f64;
    for i in 0..n {
        let mut row = 0.0;
        for k in 0..n {
            let v = gram.get(i, k);
            row += v.abs();
            frob += v * v;
        }
        gersh = gersh.max(row);
    }
    gersh.min(frob.sqrt()) + eta
}

/// Objective `½wᵀ(G + ηI)w − bᵀw` of the simplex QP.
fn objective(gram: &Mat, b: &[f64], eta: f64, w: &[f64]) -> f64 {
    let gw = matvec(gram, w);
    let mut q = 0.0;
    for i in 0..w.len() {
        q += w[i] * (gw[i] + eta * w[i]);
    }
    0.5 * q - dot(b, w)
}

/// Exact active-set polish for the bounded QP.
///
/// First-order methods identify *which* coordinates sit at their bounds quickly
/// and then spend thousands of iterations converging inside the remaining face.
/// This jumps straight to the end of that: take the coordinates of `w` strictly
/// inside the box as the free set `F`, hold the rest pegged, and solve the
/// equality-constrained QP on `F` in closed form via its KKT system
///
/// ```text
///   [G_FF  1] [w_F]   [b_F − G_FP w_P]
///   [ 1ᵀ   0] [ μ ] = [   1 − Σ w_P  ]
/// ```
///
/// i.e. `w_F = u + μv` with `u = G_FF⁻¹(b_F − G_FP w_P)`, `v = G_FF⁻¹1` and
/// `μ = (target − Σu)/Σv`. Returns `None` — leaving the caller's iterate alone —
/// unless the result is feasible *and* improves the objective, so a wrong guess
/// at the active set can never make the solve worse.
fn polish_bounded(
    gram: &Mat,
    b: &[f64],
    eta: f64,
    bounds: WeightBounds,
    w: &[f64],
) -> Option<Vec<f64>> {
    let j = w.len();
    let peg_tol = 1e-10 * (bounds.hi - bounds.lo).max(1e-12);
    let free: Vec<usize> = (0..j)
        .filter(|&i| w[i] > bounds.lo + peg_tol && w[i] < bounds.hi - peg_tol)
        .collect();
    let f = free.len();
    if f == 0 || (f == j && j > 256) {
        return None; // nothing pegged and too big to be worth a dense solve
    }
    // Free-block Gram and the right-hand side net of the pegged coordinates.
    let mut gff = Mat::zeros(f, f);
    for (a, &ia) in free.iter().enumerate() {
        for (c, &ic) in free.iter().enumerate() {
            gff.set(a, c, gram.get(ia, ic));
        }
    }
    let mut rhs = vec![0.0; f];
    let mut pegged_sum = 0.0;
    for i in 0..j {
        if w[i] <= bounds.lo + peg_tol || w[i] >= bounds.hi - peg_tol {
            pegged_sum += w[i];
        }
    }
    for (a, &ia) in free.iter().enumerate() {
        let mut acc = b[ia];
        for i in 0..j {
            let pegged = w[i] <= bounds.lo + peg_tol || w[i] >= bounds.hi - peg_tol;
            if pegged {
                acc -= gram.get(ia, i) * w[i];
            }
        }
        rhs[a] = acc;
    }
    // η enters the free block as a ridge; add a tiny relative floor so a singular
    // free block (collinear donors) still factors.
    let mut tr = 0.0;
    for a in 0..f {
        tr += gff.get(a, a);
    }
    let ridge = eta + 1e-12 * (tr / f as f64).max(1e-300);
    let chol = Cholesky::new_ridge(&gff, ridge).ok()?;
    let u = chol.solve_vec(&rhs);
    let v = chol.solve_vec(&vec![1.0; f]);
    let sum_v: f64 = v.iter().sum();
    if !sum_v.is_finite() || sum_v.abs() < 1e-300 {
        return None;
    }
    let target = 1.0 - pegged_sum;
    let mu = (target - u.iter().sum::<f64>()) / sum_v;
    let mut cand = w.to_vec();
    for (a, &ia) in free.iter().enumerate() {
        let val = u[a] + mu * v[a];
        if !val.is_finite() || val < bounds.lo - 1e-12 || val > bounds.hi + 1e-12 {
            return None; // the guessed active set was wrong
        }
        cand[ia] = val.clamp(bounds.lo, bounds.hi);
    }
    if objective(gram, b, eta, &cand) <= objective(gram, b, eta, w) {
        Some(cand)
    } else {
        None
    }
}

/// Solve the simplex QP with per-coordinate bounds
/// `lo ≤ w_i ≤ hi` (still `Σ w = 1`).
///
/// The away-step Frank–Wolfe solver above exploits the fact that the plain
/// simplex's vertices are the basis vectors — which stops being true once the
/// weights are capped, so this uses **accelerated projected gradient** (FISTA
/// with O'Donoghue–Candès gradient restart) over the exact projection
/// [`project_bounded_simplex`]. Stopping is on the Frank–Wolfe duality gap
/// `gᵀw − min_{s∈P} gᵀs ≥ f(w) − f*`, an actual optimality certificate rather
/// than a movement heuristic; the gap is evaluated every few iterations since it
/// costs an extra gradient.
///
/// Bounds that don't bind fall through to [`solve_fw`], so the default path is
/// bit-for-bit the classic solver.
pub fn solve_bounded(
    gram: &Mat,
    b: &[f64],
    eta: f64,
    bounds: WeightBounds,
    max_iter: usize,
    tol: f64,
) -> SimplexSolution {
    let j = b.len();
    if j == 0 {
        return SimplexSolution {
            w: Vec::new(),
            iters: 0,
            gap: 0.0,
        };
    }
    if !bounds.binds(j) {
        return solve_fw(gram, b, eta, max_iter, tol);
    }
    let bounds = bounds.feasible_for(j);
    let l = lipschitz_bound(gram, eta).max(1e-12);
    let step = 1.0 / l;
    // The duality gap carries the objective's units (`½wᵀGw − bᵀw`), which scale
    // with the *square* of the outcome scale — on a panel of revenue in the
    // thousands the objective is ~1e5 and an absolute 1e-12 gap is orders of
    // magnitude below what f64 can resolve there, so the loop would always run to
    // `max_iter`. Interpret `tol` relative to the problem's own scale instead:
    // `max|b|` has exactly those units, so the criterion is scale-covariant.
    let b_scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = tol * (1.0 + b_scale);

    // Start at the (always feasible) barycenter.
    let mut w = vec![1.0 / j as f64; j];
    let mut z = w.clone();
    let mut t = 1.0_f64;
    let mut gap = f64::INFINITY;
    // The gap costs an extra gradient, so check it periodically rather than
    // every step. The polish attempt costs a small Cholesky, so it is rarer
    // still — but once the active set has settled it lands on the exact optimum
    // and ends the solve, which is what keeps the bounded path from paying
    // first-order tail convergence.
    let gap_every = 5;
    let polish_every = 25;

    for k in 0..max_iter {
        let g = grad(gram, b, eta, &z);
        let trial: Vec<f64> = (0..j).map(|i| z[i] - step * g[i]).collect();
        let mut w_next = project_bounded_simplex(&trial, bounds);

        if k > 0 && k % polish_every == 0 {
            if let Some(p) = polish_bounded(gram, b, eta, bounds, &w_next) {
                w_next = p;
            }
        }

        if k % gap_every == 0 || k + 1 == max_iter {
            let gw = grad(gram, b, eta, &w_next);
            let s = lmo_bounded(&gw, bounds);
            gap = dot(&gw, &w_next) - dot(&gw, &s);
            if gap <= tol {
                return SimplexSolution {
                    w: w_next,
                    iters: k,
                    gap,
                };
            }
        }

        // Gradient restart: if the momentum direction stopped being a descent
        // direction, drop back to a plain projected-gradient step.
        let restart: f64 = (0..j)
            .map(|i| (z[i] - w_next[i]) * (w_next[i] - w[i]))
            .sum();
        let t_next = if restart > 0.0 {
            1.0
        } else {
            0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt())
        };
        let beta = if restart > 0.0 {
            0.0
        } else {
            (t - 1.0) / t_next
        };
        z = (0..j)
            .map(|i| w_next[i] + beta * (w_next[i] - w[i]))
            .collect();
        w = w_next;
        t = t_next;
    }
    SimplexSolution {
        w,
        iters: max_iter,
        gap,
    }
}

/// Synthetic-control weights with per-donor bounds — [`sc_weights`] plus a
/// `lo ≤ w_i ≤ hi` box. `y0` is `m×J` (pre-period donors), `y` length `m`.
pub fn sc_weights_bounded(y0: &Mat, y: &[f64], eta: f64, bounds: WeightBounds) -> SimplexSolution {
    if !bounds.binds(y0.cols()) {
        return sc_weights(y0, y, eta);
    }
    let gram = crate::ops::matmul::syrk_ata(y0);
    let b = crate::ops::matmul::matvec_t(y0, y);
    solve_bounded(&gram, &b, eta, bounds, 20_000, 1e-12)
}
