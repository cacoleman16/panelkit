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
}

/// Full Callaway–Sant'Anna result.
#[derive(Clone, Debug)]
pub struct CsResult {
    pub group_time: Vec<GroupTimeAtt>,
    pub event_study: Vec<AggEffect>,
    pub overall_att: f64,
    pub overall_se: f64,
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

/// Fit Callaway–Sant'Anna with a never-treated comparison group.
pub fn fit(panel: &Panel) -> CsResult {
    let n = panel.n_units();
    let t = panel.n_periods();
    let never: Vec<usize> = panel.never_treated_units();
    assert!(
        !never.is_empty(),
        "Callaway–Sant'Anna (never-treated variant) needs never-treated units"
    );

    // Cohorts with a usable base period (g >= 1).
    let cohorts: Vec<usize> = panel.cohorts().into_iter().filter(|&g| g >= 1).collect();

    // Units in each cohort.
    let cohort_units = |g: usize| -> Vec<usize> {
        (0..n)
            .filter(|&i| panel.treat_start()[i] == Some(g))
            .collect()
    };

    let nc = never.len();
    let mut group_time = Vec::new();

    for &g in &cohorts {
        let units_g = cohort_units(g);
        let ng = units_g.len();
        if ng == 0 {
            continue;
        }
        let base = g - 1;
        for period in 0..t {
            if period == base {
                continue; // base period: ATT ≡ 0 by construction
            }
            // Long differences relative to the base period.
            let dy = |i: usize| panel.outcome(i, period) - panel.outcome(i, base);

            let m_g: f64 = units_g.iter().map(|&i| dy(i)).sum::<f64>() / ng as f64;
            let m_c: f64 = never.iter().map(|&i| dy(i)).sum::<f64>() / nc as f64;
            let att = m_g - m_c;

            // Influence function over all N units (total-N scaling, consistent
            // across (g,t) so aggregations combine correctly).
            let mut influence = vec![0.0; n];
            let p_g = ng as f64 / n as f64;
            let p_c = nc as f64 / n as f64;
            for &i in &units_g {
                influence[i] = (dy(i) - m_g) / p_g;
            }
            for &i in &never {
                influence[i] -= (dy(i) - m_c) / p_c;
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

    // --- Event-study aggregation by relative time e = t − g (cohort-size weighted). ---
    let mut event_times: Vec<i64> = group_time.iter().map(|gt| gt.event_time).collect();
    event_times.sort_unstable();
    event_times.dedup();

    let cohort_size = |g: usize| cohort_units(g).len() as f64;

    let mut event_study = Vec::new();
    for &e in &event_times {
        let members: Vec<&GroupTimeAtt> =
            group_time.iter().filter(|gt| gt.event_time == e).collect();
        let total_w: f64 = members.iter().map(|gt| cohort_size(gt.cohort)).sum();
        if total_w <= 0.0 {
            continue;
        }
        let mut att = 0.0;
        let mut agg_if = vec![0.0; n];
        for gt in &members {
            let w = cohort_size(gt.cohort) / total_w;
            att += w * gt.att;
            for i in 0..n {
                agg_if[i] += w * gt.influence[i];
            }
        }
        event_study.push(AggEffect {
            key: e,
            att,
            se: se_from_if(&agg_if),
        });
    }

    // --- Overall ATT: cohort-size-weighted average of post-treatment ATT(g,t). ---
    let post: Vec<&GroupTimeAtt> = group_time.iter().filter(|gt| gt.event_time >= 0).collect();
    let total_w: f64 = post.iter().map(|gt| cohort_size(gt.cohort)).sum();
    let mut overall_att = 0.0;
    let mut overall_if = vec![0.0; n];
    if total_w > 0.0 {
        for gt in &post {
            let w = cohort_size(gt.cohort) / total_w;
            overall_att += w * gt.att;
            for i in 0..n {
                overall_if[i] += w * gt.influence[i];
            }
        }
    }
    let overall_se = se_from_if(&overall_if);

    CsResult {
        group_time,
        event_study,
        overall_att,
        overall_se,
    }
}
