//! Matrix Completion with Nuclear-Norm Minimization (Athey, Bayati,
//! Doudchenko, Imbens & Khosravi 2021), via the SoftImpute algorithm.
//!
//! The treated post-treatment entries of the outcome matrix are treated as
//! *missing*; the control entries and treated pre-period entries are observed.
//! Following the paper, the model is
//!
//! ```text
//!   Y ≈ L  +  Γ 1ᵀ  +  1 Δᵀ          (L low-rank, Γ/Δ unit & time effects)
//! ```
//!
//! with the nuclear-norm penalty on **L only** — the two-way fixed effects are
//! estimated *unpenalized* (Athey et al., §8.3). This matters in practice:
//! without the FE terms, the level component of `Y` (usually the dominant
//! singular direction for positive outcomes like revenue) gets shrunk by `λ`,
//! biasing every imputed cell by a fraction of the outcome *level* — the same
//! order as the effects being measured.
//!
//! Block-coordinate iteration:
//! 1. FE: alternating row/column means of `Y − L` over **observed** cells.
//! 2. Fill: `M = observed(Y − Γ⊕Δ) + missing(L)`.
//! 3. Threshold: `L ← SVT_λ(M)`.
//!
//! until the fitted matrix `L + Γ⊕Δ` stabilizes. `λ` is chosen by
//! cross-validation over a random hold-out of observed entries, walking the
//! grid from large to small λ with **warm starts** (Mazumder, Hastie &
//! Tibshirani's continuation) — which also removes the spurious `λ ≈ 0` fixed
//! point the cold-started iteration had (it returned "counterfactual = the
//! zero fill").

use crate::panel::Panel;
use crate::result::ScFit;
use panelkit_linalg::opt::softthresh::svt;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// Configuration for MC-NNM.
#[derive(Clone, Copy, Debug)]
pub struct McnnmConfig {
    /// Nuclear-norm penalty. If `None`, chosen by cross-validation.
    pub lambda: Option<f64>,
    /// Max SoftImpute iterations.
    pub max_iter: usize,
    /// Relative convergence tolerance (Frobenius, on the fitted matrix).
    pub tol: f64,
    /// Seed for the CV hold-out (determinism).
    pub seed: u64,
    /// Optional rank cap: when set, each SoftImpute iteration uses a fast
    /// **randomized truncated SVD** (rank `max_rank`) instead of a full SVD.
    /// A large speedup when the counterfactual is low-rank (the usual case),
    /// while staying dependency-free. `None` = exact full SVD.
    pub max_rank: Option<usize>,
}

impl Default for McnnmConfig {
    fn default() -> Self {
        McnnmConfig {
            lambda: None,
            // SoftImpute warm-starts L across iterations, so it converges in a
            // few dozen steps; 100 is a generous cap. MC-NNM is intrinsically the
            // heavy estimator (a full SVD per iteration) — keep the cap modest.
            max_iter: 100,
            tol: 1e-4,
            seed: 0,
            max_rank: None,
        }
    }
}

/// The running estimate: low-rank part plus unpenalized two-way fixed effects.
#[derive(Clone)]
struct FitState {
    l: Mat,
    /// Unit (row) effects Γ, length `n`.
    a: Vec<f64>,
    /// Time (column) effects Δ, length `t`.
    b: Vec<f64>,
}

impl FitState {
    fn zeros(n: usize, t: usize) -> FitState {
        FitState {
            l: Mat::zeros(n, t),
            a: vec![0.0; n],
            b: vec![0.0; t],
        }
    }

    /// Fitted value for cell `(i, j)`: `L[i,j] + Γ_i + Δ_j`.
    #[inline]
    fn fitted(&self, i: usize, j: usize) -> f64 {
        self.l.get(i, j) + self.a[i] + self.b[j]
    }
}

/// Re-estimate the two-way fixed effects on observed cells of `Y − L` by a few
/// passes of alternating row/column means (exact for the two-way model on a
/// near-balanced observation pattern; the missing block makes it iterative).
fn update_fe(y: &Mat, observed: &[bool], state: &mut FitState) {
    let (n, t) = y.shape();
    for _ in 0..3 {
        for i in 0..n {
            let mut s = 0.0;
            let mut c = 0usize;
            for j in 0..t {
                if observed[i + j * n] {
                    s += y.get(i, j) - state.l.get(i, j) - state.b[j];
                    c += 1;
                }
            }
            if c > 0 {
                state.a[i] = s / c as f64;
            }
        }
        for j in 0..t {
            let mut s = 0.0;
            let mut c = 0usize;
            for i in 0..n {
                if observed[i + j * n] {
                    s += y.get(i, j) - state.l.get(i, j) - state.a[i];
                    c += 1;
                }
            }
            if c > 0 {
                state.b[j] = s / c as f64;
            }
        }
    }
}

/// One SoftImpute solve for a fixed `lambda`, warm-started from `state`.
/// `observed[i + j*n]` (column-major) flags whether entry `(i, j)` is observed.
/// Solver knobs (`max_iter`, `tol`, `max_rank`, `seed`) come from `cfg`.
fn soft_impute_fe(
    y: &Mat,
    observed: &[bool],
    lambda: f64,
    cfg: &McnnmConfig,
    mut state: FitState,
) -> FitState {
    let (n, t) = y.shape();
    let mut m = Mat::zeros(n, t);
    for _ in 0..cfg.max_iter {
        // 1. Unpenalized two-way FE on the observed residual Y − L.
        update_fe(y, observed, &mut state);

        // 2. Fill: M = observed ? (Y − Γ⊕Δ) : L.
        for j in 0..t {
            for i in 0..n {
                let idx = i + j * n;
                m.as_mut_slice()[idx] = if observed[idx] {
                    y.as_slice()[idx] - state.a[i] - state.b[j]
                } else {
                    state.l.as_slice()[idx]
                };
            }
        }

        // 3. Threshold the low-rank part.
        let (l_new, _nuc) = match cfg.max_rank {
            Some(r) => panelkit_linalg::opt::softthresh::svt_truncated(&m, lambda, r, cfg.seed),
            None => svt(&m, lambda),
        };

        // Convergence on the relative Frobenius change of L (the FE step is a
        // deterministic function of L, so L stabilizing ⇒ the fit stabilizes).
        let mut num = 0.0;
        let mut den = 0.0;
        for idx in 0..n * t {
            let d = l_new.as_slice()[idx] - state.l.as_slice()[idx];
            num += d * d;
            den += state.l.as_slice()[idx] * state.l.as_slice()[idx];
        }
        state.l = l_new;
        if (den > 0.0 && (num / den).sqrt() < cfg.tol) || (den == 0.0 && num.sqrt() < cfg.tol) {
            break;
        }
    }
    // Leave the FE consistent with the final L.
    update_fe(y, observed, &mut state);
    state
}

/// Largest singular value of the FE-residualized, zero-filled matrix — anchors
/// the λ grid. (Anchoring on the raw matrix would put the whole grid at the
/// scale of the outcome *level*, which the FE terms absorb.)
fn sigma_max_fe_residual(y: &Mat, observed: &[bool]) -> f64 {
    let (n, t) = y.shape();
    let mut state = FitState::zeros(n, t);
    update_fe(y, observed, &mut state);
    let mut m = Mat::zeros(n, t);
    for j in 0..t {
        for i in 0..n {
            let idx = i + j * n;
            m.as_mut_slice()[idx] = if observed[idx] {
                y.as_slice()[idx] - state.a[i] - state.b[j]
            } else {
                0.0
            };
        }
    }
    let svd = panelkit_linalg::factor::svd::Svd::new(&m);
    svd.singular_values().first().copied().unwrap_or(0.0)
}

/// Geometric λ path from `hi` down to `lo` (inclusive), `n_grid` points.
fn lambda_path(hi: f64, lo: f64, n_grid: usize) -> Vec<f64> {
    (0..n_grid)
        .map(|k| {
            let frac = k as f64 / (n_grid as f64 - 1.0).max(1.0);
            hi * (lo / hi).powf(frac)
        })
        .collect()
}

/// Cross-validate λ over a geometric grid by holding out a random subset of
/// observed entries and minimizing held-out MSE. The grid is walked from large
/// to small λ with warm starts (continuation), so later (smaller) λs start
/// from an informative L rather than zero.
fn cv_lambda(y: &Mat, observed: &[bool], cfg: &McnnmConfig) -> f64 {
    let (n, t) = y.shape();
    let smax = sigma_max_fe_residual(y, observed);
    if smax <= 0.0 {
        return f64::MIN_POSITIVE;
    }
    // Grid: 10 points geometric from 0.5·σ_max down to 0.01·σ_max (of the
    // FE-residual spectrum, so the grid spans the low-rank component's scale).
    let grid = lambda_path(0.5 * smax, 0.01 * smax, 10);

    // Hold out ~20% of observed entries as a validation set.
    let mut rng = Xoshiro256pp::seed_from_u64(cfg.seed);
    let mut train = observed.to_vec();
    let mut val_idx = Vec::new();
    for idx in 0..n * t {
        if observed[idx] && rng.next_f64() < 0.2 {
            train[idx] = false;
            val_idx.push(idx);
        }
    }
    if val_idx.is_empty() {
        return grid[grid.len() / 2];
    }

    let mut best_lambda = grid[0];
    let mut best_err = f64::INFINITY;
    let mut state = FitState::zeros(n, t);
    for &lam in &grid {
        state = soft_impute_fe(y, &train, lam, cfg, state);
        let mut err = 0.0;
        for &idx in &val_idx {
            let (i, j) = (idx % n, idx / n);
            let d = y.as_slice()[idx] - state.fitted(i, j);
            err += d * d;
        }
        err /= val_idx.len() as f64;
        if err < best_err {
            best_err = err;
            best_lambda = lam;
        }
    }
    best_lambda
}

/// Fit MC-NNM on a block-treatment panel.
pub fn fit(panel: &Panel, cfg: McnnmConfig) -> ScFit {
    let t0 = panel
        .common_treat_time()
        .expect("MC-NNM requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Fit MC-NNM treating `t0` as the first post-period.
pub fn fit_at(panel: &Panel, t0: usize, cfg: McnnmConfig) -> ScFit {
    let y = panel.y().clone();
    let (n, t) = y.shape();

    // Observed mask (column-major): missing = treated unit in a post period.
    let mut observed = vec![true; n * t];
    for i in 0..n {
        for p in 0..t {
            if panel.is_treated(i, p) {
                observed[i + p * n] = false;
            }
        }
    }

    let lambda = cfg.lambda.unwrap_or_else(|| cv_lambda(&y, &observed, &cfg));
    // Continuation: walk a short warm-started path from a large λ down to the
    // target. Besides speed, this removes the cold-start λ ≈ 0 trivial fixed
    // point (L = 0 fills missing cells with zero and "converges" immediately).
    let smax = sigma_max_fe_residual(&y, &observed);
    let mut state = FitState::zeros(n, t);
    if smax > 0.0 && lambda < 0.5 * smax {
        for lam in lambda_path(0.5 * smax, lambda, 5) {
            state = soft_impute_fe(&y, &observed, lam, &cfg, state);
        }
    } else {
        state = soft_impute_fe(&y, &observed, lambda, &cfg, state);
    }

    let treated = panel.treated_units();
    let t_post = t - t0;

    // ATT path: per post period, average over treated units of (Y − fitted).
    let mut att_path = vec![0.0; t_post];
    let mut cf_post = vec![0.0; t_post];
    let mut treated_post = vec![0.0; t_post];
    for (pi, p) in (t0..t).enumerate() {
        let mut eff = 0.0;
        let mut cf = 0.0;
        let mut yv = 0.0;
        for &u in &treated {
            let f = state.fitted(u, p);
            eff += y.get(u, p) - f;
            cf += f;
            yv += y.get(u, p);
        }
        let inv = 1.0 / treated.len() as f64;
        att_path[pi] = eff * inv;
        cf_post[pi] = cf * inv;
        treated_post[pi] = yv * inv;
    }
    let att = att_path.iter().sum::<f64>() / t_post.max(1) as f64;

    // Pre-period fit RMSE on treated units (observed cells, fitted by L + FE).
    let mut pre_ss = 0.0;
    let mut pre_cnt = 0usize;
    for &u in &treated {
        for p in 0..t0 {
            let d = y.get(u, p) - state.fitted(u, p);
            pre_ss += d * d;
            pre_cnt += 1;
        }
    }
    let pre_rmspe = if pre_cnt > 0 {
        (pre_ss / pre_cnt as f64).sqrt()
    } else {
        0.0
    };
    let post_rmspe = {
        let m = att;
        let ss: f64 = att_path.iter().map(|r| (r - m).powi(2)).sum();
        (ss / t_post.max(1) as f64).sqrt()
    };

    ScFit {
        weights: Vec::new(),
        donor_ids: Vec::new(),
        att_path,
        att,
        counterfactual_post: cf_post,
        treated_post,
        pre_rmspe,
        post_rmspe,
    }
}
