//! Sun & Abraham (2021) interaction-weighted event study.
//!
//! Estimates the saturated two-way fixed-effects regression
//! ```text
//!   Y_it = α_i + γ_t + Σ_{g} Σ_{e≠−1} δ_{g,e}·1{cohort_i=g, t−g=e} + ε_it
//! ```
//! with never-treated units as the reference and the base relative period
//! `e = −1` omitted, then forms the interaction-weighted event-study
//! coefficients `β_e = Σ_g (cohort share at e)·δ_{g,e}`. Unit and time fixed
//! effects are absorbed by the two-way within transform (FWL); the remaining
//! interaction design is solved by QR. Standard errors are cluster-robust by
//! unit and propagated to the aggregated coefficients.

use crate::did::callaway::AggEffect;
use crate::fe::within::two_way_within;
use crate::panel::Panel;
use panelkit_linalg::factor::cholesky::Cholesky;
use panelkit_linalg::factor::qr::Qr;
use panelkit_linalg::ops::matmul::{matvec, syrk_ata};
use panelkit_linalg::Mat;

/// Sun–Abraham result.
#[derive(Clone, Debug)]
pub struct SaResult {
    pub event_study: Vec<AggEffect>,
    pub overall_att: f64,
    pub overall_se: f64,
}

/// Fit the Sun–Abraham interaction-weighted estimator.
pub fn fit(panel: &Panel) -> SaResult {
    let n = panel.n_units();
    let t = panel.n_periods();
    assert!(
        !panel.never_treated_units().is_empty(),
        "Sun–Abraham (this variant) uses never-treated units as the reference"
    );

    let cohorts: Vec<usize> = panel.cohorts().into_iter().filter(|&g| g >= 1).collect();
    let cohort_size = |g: usize| -> f64 {
        (0..n)
            .filter(|&i| panel.treat_start()[i] == Some(g))
            .count() as f64
    };

    // Enumerate interaction terms (g, e), e ≠ −1, present in the data.
    let mut terms: Vec<(usize, i64)> = Vec::new();
    for &g in &cohorts {
        for period in 0..t {
            let e = period as i64 - g as i64;
            if e == -1 {
                continue;
            }
            if !terms.contains(&(g, e)) {
                terms.push((g, e));
            }
        }
    }
    terms.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let k = terms.len();

    // Build the FWL-demeaned design: column per term, rows = flattened N×T.
    let mut x = Mat::zeros(n * t, k);
    for (kc, &(g, e)) in terms.iter().enumerate() {
        let mut ind = Mat::zeros(n, t);
        for i in 0..n {
            if panel.treat_start()[i] == Some(g) {
                let period = e + g as i64;
                if period >= 0 && (period as usize) < t {
                    ind.set(i, period as usize, 1.0);
                }
            }
        }
        let dind = two_way_within(&ind);
        // Column-major data of dind == flattened (i + t*N) which matches x rows.
        x.col_mut(kc).copy_from_slice(dind.as_slice());
    }

    // Target: flattened two-way-demeaned outcomes.
    let yt = two_way_within(panel.y());
    let target: Vec<f64> = yt.as_slice().to_vec();

    // OLS via QR.
    let qr = Qr::new(&x).expect("SA design QR");
    let delta = qr.solve_lstsq(&target);

    // Residuals.
    let fitted = matvec(&x, &delta);
    let resid: Vec<f64> = target
        .iter()
        .zip(fitted.iter())
        .map(|(a, b)| a - b)
        .collect();

    // Cluster-robust (by unit) covariance: (XᵀX)⁻¹ M (XᵀX)⁻¹,
    // M = Σ_u s_u s_uᵀ, s_u = Σ_t X[(u,t),:]·ê[(u,t)].
    let xtx = syrk_ata(&x);
    let chol = Cholesky::new_ridge(&xtx, 1e-10).expect("XtX SPD");
    // (XᵀX)⁻¹ columns.
    let mut xtx_inv = Mat::zeros(k, k);
    for j in 0..k {
        let mut e_j = vec![0.0; k];
        e_j[j] = 1.0;
        let col = chol.solve_vec(&e_j);
        xtx_inv.col_mut(j).copy_from_slice(&col);
    }
    // Meat.
    let mut meat = Mat::zeros(k, k);
    for u in 0..n {
        let mut s = vec![0.0; k];
        for period in 0..t {
            let row = u + period * n;
            let er = resid[row];
            if er == 0.0 {
                continue;
            }
            for kc in 0..k {
                s[kc] += x.get(row, kc) * er;
            }
        }
        for a in 0..k {
            if s[a] == 0.0 {
                continue;
            }
            for b in 0..k {
                meat.add_to(a, b, s[a] * s[b]);
            }
        }
    }
    // Cov = XtXinv · Meat · XtXinv  (symmetric).
    let tmp = panelkit_linalg::ops::matmul::matmul(&xtx_inv, &meat);
    let cov = panelkit_linalg::ops::matmul::matmul(&tmp, &xtx_inv);

    // Helper: variance of a linear combo aᵀδ given Cov.
    let quad = |a: &[f64]| -> f64 {
        let ca = matvec(&cov, a);
        a.iter().zip(ca.iter()).map(|(x, y)| x * y).sum::<f64>()
    };

    // Event-study aggregation: β_e = Σ_g (n_g / Σn_g) δ_{g,e}.
    let mut rel_times: Vec<i64> = terms.iter().map(|&(_, e)| e).collect();
    rel_times.sort_unstable();
    rel_times.dedup();

    let mut event_study = Vec::new();
    for &e in &rel_times {
        let total: f64 = terms
            .iter()
            .filter(|&&(_, te)| te == e)
            .map(|&(g, _)| cohort_size(g))
            .sum();
        if total <= 0.0 {
            continue;
        }
        let mut a = vec![0.0; k];
        let mut att = 0.0;
        for (kc, &(g, te)) in terms.iter().enumerate() {
            if te == e {
                let w = cohort_size(g) / total;
                a[kc] = w;
                att += w * delta[kc];
            }
        }
        event_study.push(AggEffect {
            key: e,
            att,
            se: quad(&a).max(0.0).sqrt(),
        });
    }

    // Overall ATT: cohort-size-weighted average of δ_{g,e} over e ≥ 0.
    let total_post: f64 = terms
        .iter()
        .filter(|&&(_, e)| e >= 0)
        .map(|&(g, _)| cohort_size(g))
        .sum();
    let mut a_overall = vec![0.0; k];
    let mut overall_att = 0.0;
    if total_post > 0.0 {
        for (kc, &(g, e)) in terms.iter().enumerate() {
            if e >= 0 {
                let w = cohort_size(g) / total_post;
                a_overall[kc] = w;
                overall_att += w * delta[kc];
            }
        }
    }
    let overall_se = quad(&a_overall).max(0.0).sqrt();

    SaResult {
        event_study,
        overall_att,
        overall_se,
    }
}
