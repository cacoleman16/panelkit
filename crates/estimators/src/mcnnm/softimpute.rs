//! Matrix Completion with Nuclear-Norm Minimization (Athey, Bayati,
//! Doudchenko, Imbens & Khosravi 2021), via the SoftImpute algorithm.
//!
//! The treated post-treatment entries of the outcome matrix are treated as
//! *missing*; the control entries and treated pre-period entries are observed.
//! We estimate a low-rank counterfactual `L` by iterating:
//!
//! 1. Fill: `M = observed(Y) + missing(L)` (impute missing cells with the
//!    current estimate).
//! 2. Threshold: `L ← SVT_λ(M)` (soft-threshold the singular values by `λ`).
//!
//! until `L` stabilizes. The treatment effect on each treated post entry is
//! `Y − L`. The regularization `λ` is chosen by cross-validation over a random
//! hold-out of observed entries (deterministic given the seed).

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
    /// Relative convergence tolerance (Frobenius).
    pub tol: f64,
    /// Seed for the CV hold-out (determinism).
    pub seed: u64,
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
        }
    }
}

/// One SoftImpute solve for a fixed `lambda`. `observed[i + j*n]` (column-major)
/// flags whether entry `(i, j)` is observed. Returns the low-rank estimate `L`.
fn soft_impute(y: &Mat, observed: &[bool], lambda: f64, max_iter: usize, tol: f64) -> Mat {
    let (n, t) = y.shape();
    let mut l = Mat::zeros(n, t);
    let mut m = Mat::zeros(n, t);
    for _ in 0..max_iter {
        // Fill: M = observed ? Y : L.
        for idx in 0..n * t {
            m.as_mut_slice()[idx] = if observed[idx] {
                y.as_slice()[idx]
            } else {
                l.as_slice()[idx]
            };
        }
        let (l_new, _nuc) = svt(&m, lambda);
        // Convergence on relative Frobenius change.
        let mut num = 0.0;
        let mut den = 0.0;
        for idx in 0..n * t {
            let d = l_new.as_slice()[idx] - l.as_slice()[idx];
            num += d * d;
            den += l.as_slice()[idx] * l.as_slice()[idx];
        }
        l = l_new;
        if den > 0.0 && (num / den).sqrt() < tol {
            break;
        }
        if den == 0.0 && num.sqrt() < tol {
            break;
        }
    }
    l
}

/// Largest singular value of the observed-mean-filled matrix — anchors the λ grid.
fn sigma_max_filled(y: &Mat, observed: &[bool]) -> f64 {
    let (n, t) = y.shape();
    // Mean of observed entries.
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for idx in 0..n * t {
        if observed[idx] {
            sum += y.as_slice()[idx];
            cnt += 1;
        }
    }
    let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    let mut m = Mat::zeros(n, t);
    for idx in 0..n * t {
        m.as_mut_slice()[idx] = if observed[idx] {
            y.as_slice()[idx]
        } else {
            mean
        };
    }
    let svd = panelkit_linalg::factor::svd::Svd::new(&m);
    svd.singular_values().first().copied().unwrap_or(0.0)
}

/// Cross-validate λ over a geometric grid by holding out a random subset of
/// observed entries and minimizing held-out MSE.
fn cv_lambda(y: &Mat, observed: &[bool], cfg: &McnnmConfig) -> f64 {
    let (n, t) = y.shape();
    let smax = sigma_max_filled(y, observed);
    if smax <= 0.0 {
        return 0.0;
    }
    // Grid: 10 points geometric from 0.5·σ_max down to 0.01·σ_max.
    let n_grid = 10;
    let hi = 0.5 * smax;
    let lo = 0.01 * smax;
    let grid: Vec<f64> = (0..n_grid)
        .map(|k| {
            let frac = k as f64 / (n_grid as f64 - 1.0);
            hi * (lo / hi).powf(frac)
        })
        .collect();

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
    for &lam in &grid {
        let l = soft_impute(y, &train, lam, cfg.max_iter, cfg.tol);
        let mut err = 0.0;
        for &idx in &val_idx {
            let d = y.as_slice()[idx] - l.as_slice()[idx];
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
    let l = soft_impute(&y, &observed, lambda, cfg.max_iter, cfg.tol);

    let treated = panel.treated_units();
    let t_post = t - t0;

    // ATT path: per post period, average over treated units of (Y − L).
    let mut att_path = vec![0.0; t_post];
    let mut cf_post = vec![0.0; t_post];
    let mut treated_post = vec![0.0; t_post];
    for (pi, p) in (t0..t).enumerate() {
        let mut eff = 0.0;
        let mut cf = 0.0;
        let mut yv = 0.0;
        for &u in &treated {
            eff += y.get(u, p) - l.get(u, p);
            cf += l.get(u, p);
            yv += y.get(u, p);
        }
        let inv = 1.0 / treated.len() as f64;
        att_path[pi] = eff * inv;
        cf_post[pi] = cf * inv;
        treated_post[pi] = yv * inv;
    }
    let att = att_path.iter().sum::<f64>() / t_post.max(1) as f64;

    // Pre-period fit RMSE on treated units (observed cells imputed by L).
    let mut pre_ss = 0.0;
    let mut pre_cnt = 0usize;
    for &u in &treated {
        for p in 0..t0 {
            let d = y.get(u, p) - l.get(u, p);
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
