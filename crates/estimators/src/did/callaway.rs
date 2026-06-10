//! Callaway & Sant'Anna (2021) group-time ATTs with a never-treated comparison
//! group, plus event-study and overall aggregations.
//!
//! For cohort `g` (units first treated at period `g`) and period `t`, using the
//! base period `g−1`:
//! ```text
//!   ATT(g,t) = E[Y_t − Y_{g−1} | G = g] − E[Y_t − Y_{g−1} | never-treated]
//! ```
//! a difference of two long-difference sample means. Each ATT(g,t) carries a
//! unit-level **influence function** so that any linear aggregation (event-study
//! by relative time, or the overall ATT) gets a correct standard error, and so
//! the multiplier bootstrap can resample the influence functions directly.
//!
//! This is the "simple" (no-covariate, never-treated) variant; the not-yet-
//! treated and doubly-robust variants are clean extensions of the same scaffold.

use crate::panel::Panel;
use panelkit_linalg::solve::ols;
use panelkit_linalg::Mat;

/// Fit a regression-adjustment model: OLS of the long-difference `dy` on
/// `[1, X]` among the comparison units. Returns `Some(beta)` (length `K+1`) when
/// the panel has covariates and the system is solvable, else `None` (→ no
/// adjustment, the simple estimator).
fn covariate_beta(panel: &Panel, dy: &dyn Fn(usize) -> f64, comp: &[usize]) -> Option<Vec<f64>> {
    let x = panel.covariates()?;
    let k = x.cols();
    if comp.len() <= k + 1 {
        return None; // too few controls to fit the regression
    }
    // Design [1, X] for comparison units; target = dy(comp).
    let mut design = Mat::zeros(comp.len(), k + 1);
    let mut target = vec![0.0; comp.len()];
    for (r, &i) in comp.iter().enumerate() {
        design.set(r, 0, 1.0);
        for c in 0..k {
            design.set(r, c + 1, x.get(i, c));
        }
        target[r] = dy(i);
    }
    ols(&design, &target).ok()
}

/// Predicted long-difference for unit `i` from a regression-adjustment `beta`
/// (`[1, X_i]·beta`); 0 when there is no adjustment.
fn predict(panel: &Panel, beta: &Option<Vec<f64>>, i: usize) -> f64 {
    match (beta, panel.covariates()) {
        (Some(b), Some(x)) => {
            let mut v = b[0];
            for c in 0..x.cols() {
                v += b[c + 1] * x.get(i, c);
            }
            v
        }
        _ => 0.0,
    }
}

/// A single group-time average treatment effect.
#[derive(Clone, Debug)]
pub struct GroupTimeAtt {
    pub cohort: usize,
    pub period: usize,
    /// Relative event time `t − g`.
    pub event_time: i64,
    pub att: f64,
    pub se: f64,
    /// Unit-level influence function (length = N), enabling aggregation.
    pub influence: Vec<f64>,
}

/// An aggregated effect with its standard error.
#[derive(Clone, Debug)]
pub struct AggEffect {
    pub key: i64,
    pub att: f64,
    pub se: f64,
    /// Unit-level influence function of the aggregate (length N, total-N
    /// scaling, including the weight-estimation term) — empty when the
    /// producing estimator does not expose one (e.g. Sun-Abraham). Feeds the
    /// multiplier bootstrap / uniform bands.
    pub influence: Vec<f64>,
}

/// Full Callaway–Sant'Anna result.
#[derive(Clone, Debug)]
pub struct CsResult {
    pub group_time: Vec<GroupTimeAtt>,
    pub event_study: Vec<AggEffect>,
    /// Per-cohort aggregation: `key` = cohort g, `att` = the average of that
    /// cohort's post-treatment ATT(g,t).
    pub group_study: Vec<AggEffect>,
    /// "Simple" overall ATT: cohort-size-weighted average over post (g,t) cells.
    pub overall_att: f64,
    pub overall_se: f64,
    /// Influence function of the simple overall ATT (length N).
    pub overall_influence: Vec<f64>,
    /// "Group" overall ATT (C&S's headline recommendation): cohort-size-weighted
    /// average of the per-cohort ATT_g — every cohort's exposure counts equally
    /// per unit, regardless of how many post periods it has.
    pub overall_group_att: f64,
    pub overall_group_se: f64,
}

/// SE of an influence vector: `sqrt((1/N²) Σ_i IF_i²)`.
fn se_from_if(influence: &[f64]) -> f64 {
    let n = influence.len();
    if n == 0 {
        return 0.0;
    }
    let ss: f64 = influence.iter().map(|v| v * v).sum();
    (ss / (n as f64 * n as f64)).sqrt()
}

/// Comparison ("control") group for the group-time ATTs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlGroup {
    /// Use only never-treated units as controls (requires some never-treated).
    NeverTreated,
    /// Use not-yet-treated units (never-treated + cohorts treated strictly after
    /// the periods involved). Larger control pool; valid without never-treated.
    NotYetTreated,
}

/// Fit Callaway–Sant'Anna with a never-treated comparison group.
pub fn fit(panel: &Panel) -> CsResult {
    fit_with(panel, ControlGroup::NeverTreated)
}

/// Fit Callaway–Sant'Anna with an explicit comparison group.
pub fn fit_with(panel: &Panel, control: ControlGroup) -> CsResult {
    fit_with_anticipation(panel, control, 0)
}

/// Fit Callaway–Sant'Anna allowing `anticipation` periods of treatment
/// anticipation: the base period moves to `g − 1 − anticipation`, so units may
/// respond up to `anticipation` periods before formal adoption without
/// contaminating the baseline. Event-study entries at `e ∈ [-anticipation, 0)`
/// then measure the anticipation response; the overall ATT still aggregates
/// `e ≥ 0` only.
pub fn fit_with_anticipation(
    panel: &Panel,
    control: ControlGroup,
    anticipation: usize,
) -> CsResult {
    let n = panel.n_units();
    let t = panel.n_periods();
    let never: Vec<usize> = panel.never_treated_units();
    if control == ControlGroup::NeverTreated {
        assert!(
            !never.is_empty(),
            "Callaway–Sant'Anna (never-treated variant) needs never-treated units"
        );
    }

    // Cohorts with a usable base period (g − 1 − anticipation >= 0).
    let cohorts: Vec<usize> = panel
        .cohorts()
        .into_iter()
        .filter(|&g| g > anticipation)
        .collect();

    // Units in each cohort.
    let cohort_units = |g: usize| -> Vec<usize> {
        (0..n)
            .filter(|&i| panel.treat_start()[i] == Some(g))
            .collect()
    };

    // Comparison set for cohort `g` at evaluation `period` (base = g-1-a):
    // - NeverTreated: the never-treated units.
    // - NotYetTreated: units whose own anticipation-adjusted start `s - a`
    //   falls strictly after BOTH the base and the evaluation period, and not
    //   in cohort g.
    let comparison = |g: usize, period: usize| -> Vec<usize> {
        let base = g - 1 - anticipation;
        match control {
            ControlGroup::NeverTreated => never.clone(),
            ControlGroup::NotYetTreated => {
                let cutoff = base.max(period);
                (0..n)
                    .filter(|&i| match panel.treat_start()[i] {
                        None => true,
                        Some(s) => s.saturating_sub(anticipation) > cutoff && s != g,
                    })
                    .collect()
            }
        }
    };

    let mut group_time = Vec::new();

    for &g in &cohorts {
        let units_g = cohort_units(g);
        let ng = units_g.len();
        if ng == 0 {
            continue;
        }
        let base = g - 1 - anticipation;
        for period in 0..t {
            if period == base {
                continue; // base period: ATT ≡ 0 by construction
            }
            let comp = comparison(g, period);
            let nc = comp.len();
            if nc == 0 {
                continue; // no valid control for this (g, t)
            }
            // Long differences relative to the base period.
            let dy = |i: usize| panel.outcome(i, period) - panel.outcome(i, base);

            // Covariate adjustment (regression-adjustment / outcome-regression):
            // regress ΔY on [1, X] among the comparison group and subtract the
            // fitted value, so e_i is the covariate-residualized change. With no
            // covariates this collapses to e_i = ΔY_i (the simple estimator).
            let beta = covariate_beta(panel, &dy, &comp);
            let e = |i: usize| dy(i) - predict(panel, &beta, i);

            let m_g: f64 = units_g.iter().map(|&i| e(i)).sum::<f64>() / ng as f64;
            let m_c: f64 = comp.iter().map(|&i| e(i)).sum::<f64>() / nc as f64;
            let att = m_g - m_c;

            // Influence function over all N units (total-N scaling, consistent
            // across (g,t) so aggregations combine correctly).
            let mut influence = vec![0.0; n];
            let p_g = ng as f64 / n as f64;
            let p_c = nc as f64 / n as f64;
            for &i in &units_g {
                influence[i] = (e(i) - m_g) / p_g;
            }
            for &i in &comp {
                influence[i] -= (e(i) - m_c) / p_c;
            }
            // First-step correction for the estimated OLS coefficients: the
            // ATT depends on β̂ through both group means, with derivative
            // −(X̄₁_g − X̄₁_c). β̂'s own influence (comparison units only) is
            // Q_c⁻¹·X̃ᵢ·eᵢ / p_c, so each comparison unit picks up
            //   −(X̄₁_g − X̄₁_c)ᵀ Q_c⁻¹ X̃ᵢ eᵢ / p_c.
            // Without it, covariate-adjusted SEs ignore that β̂ is estimated
            // (measured ~35% understated in a confounded DGP). No propensity
            // model is involved — this is the outcome-regression term only.
            if beta.is_some() {
                if let Some(x) = panel.covariates() {
                    let k = x.cols();
                    let xt = |i: usize, c: usize| if c == 0 { 1.0 } else { x.get(i, c - 1) };
                    // Q_c = (1/n_c) Σ_comp X̃ X̃ᵀ  ((k+1)×(k+1)).
                    let mut q = Mat::zeros(k + 1, k + 1);
                    for &i in &comp {
                        for a in 0..=k {
                            for b in 0..=k {
                                q.add_to(a, b, xt(i, a) * xt(i, b) / nc as f64);
                            }
                        }
                    }
                    // d = X̄₁_g − X̄₁_c (intercept components cancel).
                    let mut d = vec![0.0; k + 1];
                    for c in 1..=k {
                        let xg: f64 =
                            units_g.iter().map(|&i| x.get(i, c - 1)).sum::<f64>() / ng as f64;
                        let xc: f64 =
                            comp.iter().map(|&i| x.get(i, c - 1)).sum::<f64>() / nc as f64;
                        d[c] = xg - xc;
                    }
                    if let Ok(chol) =
                        panelkit_linalg::factor::cholesky::Cholesky::new_ridge(&q, 1e-12)
                    {
                        let v = chol.solve_vec(&d); // Q_c⁻¹ (X̄_g − X̄_c)
                        for &i in &comp {
                            let vx: f64 = (0..=k).map(|c| v[c] * xt(i, c)).sum();
                            influence[i] -= vx * e(i) / p_c;
                        }
                    }
                }
            }

            group_time.push(GroupTimeAtt {
                cohort: g,
                period,
                event_time: period as i64 - g as i64,
                att,
                se: se_from_if(&influence),
                influence,
            });
        }
    }

    // --- Aggregations (cohort-size weighted) with estimated-weight correction. ---
    //
    // The weights ŵ_m = n_{g_m}/Σ n_{g'} are themselves *estimated* (cohort
    // membership is random in C&S's sampling framework), so the aggregate's
    // influence function needs the weight-estimation term
    //     Σ_m ATT_m · dŵ_m(i),
    //     dŵ_m(i) = ψ_{g_m}(i)/P − (p_{g_m}/P²)·Σ_{m'} ψ_{g_{m'}}(i),
    //     ψ_g(i) = 1{G_i = g} − p_g,  p_g = n_g/n,  P = Σ_m p_{g_m}
    // (the `wif` of Callaway's `did` package). Omitting it understates the
    // aggregated SEs whenever effects are heterogeneous across cohorts —
    // measured ~25% on a random-cohort DGP. With homogeneous effects the term
    // is identically zero (Σŵ ≡ 1), so those SEs are unchanged.
    let cohort_of: Vec<Option<usize>> = panel.treat_start().to_vec();
    let aggregate = |members: &[&GroupTimeAtt]| -> Option<AggEffect> {
        let cohort_size = |g: usize| cohort_units(g).len() as f64;
        let p: Vec<f64> = members
            .iter()
            .map(|m| cohort_size(m.cohort) / n as f64)
            .collect();
        let cap_p: f64 = p.iter().sum();
        if cap_p <= 0.0 {
            return None;
        }
        let mut att = 0.0;
        let mut agg_if = vec![0.0; n];
        for (m, gt) in members.iter().enumerate() {
            let w = p[m] / cap_p;
            att += w * gt.att;
            for i in 0..n {
                agg_if[i] += w * gt.influence[i];
            }
        }
        // Weight-estimation (wif) term.
        for (i, agg) in agg_if.iter_mut().enumerate() {
            let mut sum_psi = 0.0;
            for (m, gt) in members.iter().enumerate() {
                let ind = if cohort_of[i] == Some(gt.cohort) {
                    1.0
                } else {
                    0.0
                };
                sum_psi += ind - p[m];
            }
            for (m, gt) in members.iter().enumerate() {
                let ind = if cohort_of[i] == Some(gt.cohort) {
                    1.0
                } else {
                    0.0
                };
                let psi_m = ind - p[m];
                let dw = psi_m / cap_p - (p[m] / (cap_p * cap_p)) * sum_psi;
                *agg += gt.att * dw;
            }
        }
        Some(AggEffect {
            key: 0,
            att,
            se: se_from_if(&agg_if),
            influence: agg_if,
        })
    };

    // Event study by relative time e = t − g.
    let mut event_times: Vec<i64> = group_time.iter().map(|gt| gt.event_time).collect();
    event_times.sort_unstable();
    event_times.dedup();

    let mut event_study = Vec::new();
    for &e in &event_times {
        let members: Vec<&GroupTimeAtt> =
            group_time.iter().filter(|gt| gt.event_time == e).collect();
        if let Some(mut agg) = aggregate(&members) {
            agg.key = e;
            event_study.push(agg);
        }
    }

    // Overall ATT ("simple"): cohort-size-weighted average over post (g,t) cells.
    let post: Vec<&GroupTimeAtt> = group_time.iter().filter(|gt| gt.event_time >= 0).collect();
    let (overall_att, overall_se, overall_influence) = match aggregate(&post) {
        Some(agg) => (agg.att, agg.se, agg.influence),
        None => (0.0, 0.0, Vec::new()),
    };

    // Per-cohort ("group") aggregation: ATT_g = average of cohort g's post
    // cells. Within a cohort the member weights are equal and the wif term
    // vanishes (the shares are identical), so `aggregate` yields the fixed-
    // weight average with the correct IF.
    let mut group_study: Vec<AggEffect> = Vec::new();
    for &g in &cohorts {
        let cells: Vec<&GroupTimeAtt> = group_time
            .iter()
            .filter(|gt| gt.cohort == g && gt.event_time >= 0)
            .collect();
        if let Some(mut agg) = aggregate(&cells) {
            agg.key = g as i64;
            group_study.push(agg);
        }
    }
    // Overall "group" ATT (C&S's recommended headline): cohort-size-weighted
    // average of the ATT_g — feed the per-cohort aggregates back through the
    // same machinery so the estimated cohort shares contribute their wif term.
    let group_members: Vec<GroupTimeAtt> = group_study
        .iter()
        .map(|agg| GroupTimeAtt {
            cohort: agg.key as usize,
            period: 0,
            event_time: 0,
            att: agg.att,
            se: agg.se,
            influence: agg.influence.clone(),
        })
        .collect();
    let member_refs: Vec<&GroupTimeAtt> = group_members.iter().collect();
    let (overall_group_att, overall_group_se) = match aggregate(&member_refs) {
        Some(agg) => (agg.att, agg.se),
        None => (0.0, 0.0),
    };

    CsResult {
        group_time,
        event_study,
        group_study,
        overall_att,
        overall_se,
        overall_influence,
        overall_group_att,
        overall_group_se,
    }
}
