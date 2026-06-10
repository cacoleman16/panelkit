//! DiD family correctness, including the headline modern-DiD result: under
//! staggered adoption with heterogeneous effects, TWFE is biased while
//! Callaway–Sant'Anna and Sun–Abraham recover the truth.

#![allow(clippy::needless_range_loop)]

use panelkit_estimators::did::{
    bacon_decompose, fit_callaway, fit_callaway_with, fit_callaway_with_anticipation, fit_sunab,
    fit_twfe, BaconKind, ControlGroup,
};
use panelkit_estimators::Panel;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// Staggered-adoption panel. Three groups: never-treated, early cohort (treated
/// at `g1`) and late cohort (treated at `g2`). Each treated unit gets a constant
/// (per-cohort) effect that turns on at adoption. With heterogeneous effects
/// across cohorts and staggered timing, TWFE's "forbidden comparisons" bias the
/// estimate; C&S/SA do not.
fn staggered_panel(eff_early: f64, eff_late: f64, seed: u64) -> (Panel, f64) {
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    let per_group = 25usize;
    let n = per_group * 3;
    let t = 16usize;
    let g1 = 5usize;
    let g2 = 10usize;

    let unit_fe: Vec<f64> = (0..n).map(|_| 3.0 + rng.next_normal()).collect();
    let mut time_fe = vec![0.0; t];
    let mut acc = 0.0;
    for v in time_fe.iter_mut() {
        acc += 0.05 * rng.next_normal();
        *v = acc;
    }

    let mut starts = vec![None; n];
    let mut y = Mat::zeros(n, t);
    // count exposed unit-periods to compute the true average post-treatment ATT
    let mut eff_sum = 0.0;
    let mut eff_cnt = 0.0;
    for i in 0..n {
        let group = i / per_group; // 0 never, 1 early, 2 late
        let (start, eff) = match group {
            1 => (Some(g1), eff_early),
            2 => (Some(g2), eff_late),
            _ => (None, 0.0),
        };
        starts[i] = start;
        for p in 0..t {
            let mut v = unit_fe[i] + time_fe[p] + 0.1 * rng.next_normal();
            if let Some(s) = start {
                if p >= s {
                    v += eff;
                    eff_sum += eff;
                    eff_cnt += 1.0;
                }
            }
            y.set(i, p, v);
        }
    }
    let true_att = eff_sum / eff_cnt; // simple average effect on the treated
    (Panel::new(y, starts), true_att)
}

#[test]
fn twfe_static_recovers_homogeneous_effect() {
    // With a SINGLE common effect and timing, TWFE is unbiased.
    let (panel, true_att) = staggered_panel(2.0, 2.0, 1);
    let fit = fit_twfe(&panel);
    assert!(
        (fit.att - true_att).abs() < 0.2,
        "TWFE {} vs true {}",
        fit.att,
        true_att
    );
    assert!(fit.se > 0.0 && fit.se < 1.0);
}

#[test]
fn callaway_recovers_heterogeneous_staggered() {
    // Strongly heterogeneous effects (early=1, late=5) + staggered timing.
    let (panel, true_att) = staggered_panel(1.0, 5.0, 2);
    let cs = fit_callaway(&panel);
    assert!(
        (cs.overall_att - true_att).abs() < 0.4,
        "C&S overall {} vs true {}",
        cs.overall_att,
        true_att
    );
    assert!(cs.overall_se > 0.0);
    // Pre-treatment event-study coefficients (e < -1) should be ~0.
    for eff in &cs.event_study {
        if eff.key < -1 {
            assert!(
                eff.att.abs() < 0.5,
                "C&S pre-trend at e={} is {}",
                eff.key,
                eff.att
            );
        }
    }
}

#[test]
fn sunab_recovers_heterogeneous_staggered() {
    let (panel, true_att) = staggered_panel(1.0, 5.0, 3);
    let sa = fit_sunab(&panel);
    assert!(
        (sa.overall_att - true_att).abs() < 0.4,
        "SA overall {} vs true {}",
        sa.overall_att,
        true_att
    );
    assert!(sa.overall_se > 0.0);
}

#[test]
fn twfe_biased_but_cs_sa_correct_under_heterogeneity() {
    // The headline test. Big effect gap across cohorts → TWFE is contaminated
    // by forbidden comparisons (already-treated used as controls), while C&S and
    // SA recover the true average effect.
    let (panel, true_att) = staggered_panel(1.0, 8.0, 4);
    let twfe = fit_twfe(&panel).att;
    let cs = fit_callaway(&panel).overall_att;
    let sa = fit_sunab(&panel).overall_att;

    let twfe_err = (twfe - true_att).abs();
    let cs_err = (cs - true_att).abs();
    let sa_err = (sa - true_att).abs();

    // C&S and SA should be close to truth...
    assert!(cs_err < 0.5, "C&S err {cs_err} (cs={cs}, true={true_att})");
    assert!(sa_err < 0.5, "SA err {sa_err} (sa={sa}, true={true_att})");
    // ...and clearly better than TWFE, which is meaningfully biased here.
    assert!(
        twfe_err > cs_err,
        "expected TWFE ({twfe}) more biased than C&S ({cs}); true={true_att}"
    );
}

#[test]
fn callaway_not_yet_treated_recovers_truth() {
    let (panel, true_att) = staggered_panel(1.0, 5.0, 7);
    let cs = fit_callaway_with(&panel, ControlGroup::NotYetTreated);
    assert!(
        (cs.overall_att - true_att).abs() < 0.4,
        "C&S(not-yet) overall {} vs true {}",
        cs.overall_att,
        true_att
    );
}

#[test]
fn callaway_not_yet_treated_works_without_never_treated() {
    // A panel with NO never-treated units: only the not-yet-treated comparison
    // is usable. (Two cohorts, everyone eventually treated.)
    let mut rng = Xoshiro256pp::seed_from_u64(8);
    let per = 30usize;
    let n = per * 2;
    let t = 16usize;
    let (g1, g2) = (5usize, 11usize);
    let ufe: Vec<f64> = (0..n).map(|_| 3.0 + rng.next_normal()).collect();
    let mut tfe = vec![0.0; t];
    let mut acc = 0.0;
    for v in tfe.iter_mut() {
        acc += 0.05 * rng.next_normal();
        *v = acc;
    }
    let mut starts = vec![None; n];
    let mut y = Mat::zeros(n, t);
    let (e1, e2) = (2.0_f64, 2.0_f64); // homogeneous so the simple average ATT is 2.0
    for i in 0..n {
        let (s, e) = if i < per { (g1, e1) } else { (g2, e2) };
        starts[i] = Some(s);
        for p in 0..t {
            let mut v = ufe[i] + tfe[p] + 0.05 * rng.next_normal();
            if p >= s {
                v += e;
            }
            y.set(i, p, v);
        }
    }
    let panel = Panel::new(y, starts);
    // never-treated variant would panic; not-yet-treated must work.
    let cs = fit_callaway_with(&panel, ControlGroup::NotYetTreated);
    assert!(
        (cs.overall_att - 2.0).abs() < 0.4,
        "C&S(not-yet) without never-treated: {} vs 2.0",
        cs.overall_att
    );
}

#[test]
fn covariate_adjustment_reduces_confounding_bias() {
    // Confound: each unit's untreated trend slope ∝ a covariate x, and treated
    // units have systematically higher x than controls. Plain C&S attributes the
    // steeper treated trend to the treatment (biased up); regression-adjusting on
    // x removes it.
    let mut rng = Xoshiro256pp::seed_from_u64(2027);
    let per = 40usize;
    let n = per * 2; // group 0 = control (low x), group 1 = treated cohort (high x)
    let t = 16usize;
    let g = 8usize;
    let true_eff = 2.0;

    let mut x = vec![0.0; n];
    let mut starts = vec![None; n];
    let mut y = Mat::zeros(n, t);
    for i in 0..n {
        let treated = i >= per;
        x[i] = if treated { 2.0 } else { 0.0 } + 0.5 * rng.next_normal();
        let unit_fe = 5.0 + rng.next_normal();
        let slope = 0.3 * x[i]; // trend depends on covariate → confound
        if treated {
            starts[i] = Some(g);
        }
        for p in 0..t {
            let mut v = unit_fe + slope * p as f64 + 0.05 * rng.next_normal();
            if treated && p >= g {
                v += true_eff;
            }
            y.set(i, p, v);
        }
    }
    let xmat = Mat::from_col_vec(&x); // N×1 covariate
    let panel = Panel::new(y, starts).with_covariates(xmat);

    // Covariates are attached, so this fit uses regression adjustment; the
    // simple comparison comes from a covariate-free copy below.
    let adjusted = fit_callaway_with(&panel, ControlGroup::NeverTreated).overall_att;
    let panel_nocov = {
        let mut yy = Mat::zeros(n, t);
        for i in 0..n {
            for p in 0..t {
                yy.set(i, p, panel.outcome(i, p));
            }
        }
        let mut st = vec![None; n];
        for i in per..n {
            st[i] = Some(g);
        }
        Panel::new(yy, st)
    };
    let simple_nocov = fit_callaway_with(&panel_nocov, ControlGroup::NeverTreated).overall_att;

    let err_adj = (adjusted - true_eff).abs();
    let err_simple = (simple_nocov - true_eff).abs();
    assert!(
        err_adj < err_simple,
        "covariate adjustment should reduce bias: adj err {err_adj} vs simple err {err_simple} (adj={adjusted}, simple={simple_nocov}, true={true_eff})"
    );
    assert!(
        err_adj < 0.5,
        "covariate-adjusted estimate {adjusted} not close to {true_eff}"
    );
}

#[test]
fn bacon_decomposition_reproduces_twfe() {
    // The decomposition's weighted average of 2x2 estimates must equal the TWFE
    // coefficient — the strongest correctness check.
    let (panel, _) = staggered_panel(1.0, 8.0, 4);
    let twfe = fit_twfe(&panel).att;
    let bacon = bacon_decompose(&panel);
    assert!(
        (bacon.twfe - twfe).abs() < 1e-9,
        "Bacon Σwβ {} != TWFE {}",
        bacon.twfe,
        twfe
    );
    // Weights sum to 1.
    let wsum: f64 = bacon.components.iter().map(|c| c.weight).sum();
    assert!((wsum - 1.0).abs() < 1e-9);
}

#[test]
fn bacon_flags_forbidden_comparison_weight() {
    // With staggered timing and a never-treated group there are forbidden
    // (later-vs-earlier) comparisons carrying positive weight — the source of
    // TWFE's bias under heterogeneity.
    let (panel, _) = staggered_panel(1.0, 8.0, 4);
    let bacon = bacon_decompose(&panel);
    assert!(
        bacon.forbidden_weight > 0.0,
        "expected positive forbidden-comparison weight"
    );
    // There is at least one forbidden component.
    assert!(bacon
        .components
        .iter()
        .any(|c| c.kind == BaconKind::LaterVsEarlierForbidden));
}

// ---------------------------------------------------------------------------
// Audit regressions: always-treated contamination (SA), estimated-weight and
// estimated-β terms in the C&S influence functions.
// ---------------------------------------------------------------------------

#[test]
fn sunab_excludes_always_treated_units() {
    // Always-treated units (cohort 0) have no reference period; leaving them
    // in pooled them into the never-treated reference and contaminated every
    // event-study coefficient (spurious pre-trends, biased dynamics).
    let (n, t) = (10usize, 12usize);
    let mut y = Mat::zeros(n, t);
    let mut starts: Vec<Option<usize>> = vec![None; n];
    for p in 0..t {
        // 4 never-treated: flat baselines.
        for i in 0..4 {
            y.set(i, p, 1.0 + i as f64);
        }
        // 3 always-treated (cohort 0) with a strongly trending treated path.
        for i in 4..7 {
            y.set(i, p, 5.0 + 0.5 * p as f64);
            starts[i] = Some(0);
        }
        // 3 treated at g = 6 with constant effect tau = 2.
        for i in 7..10 {
            let eff = if p >= 6 { 2.0 } else { 0.0 };
            y.set(i, p, 2.0 + (i as f64) * 0.1 + eff);
            starts[i] = Some(6);
        }
    }
    let full = Panel::new(y.clone(), starts.clone());

    // Reference: the same panel with the always-treated rows dropped by hand.
    let keep: Vec<usize> = (0..n).filter(|&i| starts[i] != Some(0)).collect();
    let mut y2 = Mat::zeros(keep.len(), t);
    let mut starts2 = Vec::new();
    for (r, &u) in keep.iter().enumerate() {
        for p in 0..t {
            y2.set(r, p, y.get(u, p));
        }
        starts2.push(starts[u]);
    }
    let dropped = Panel::new(y2, starts2);

    let sa_full = fit_sunab(&full);
    let sa_drop = fit_sunab(&dropped);
    assert!(
        (sa_full.overall_att - sa_drop.overall_att).abs() < 1e-10,
        "always-treated units leaked into the SA fit: {} vs {}",
        sa_full.overall_att,
        sa_drop.overall_att
    );
    // And on this noiseless design the truth is exactly 2.
    assert!(
        (sa_full.overall_att - 2.0).abs() < 1e-8,
        "SA overall {} != 2.0",
        sa_full.overall_att
    );
    for e in &sa_full.event_study {
        let want = if e.key >= 0 { 2.0 } else { 0.0 };
        assert!(
            (e.att - want).abs() < 1e-8,
            "SA event e={} att {} != {}",
            e.key,
            e.att,
            want
        );
    }
}

#[test]
fn cs_overall_ci_covers_under_random_cohorts_and_heterogeneous_effects() {
    // C&S's sampling framework treats cohort membership as random, so the
    // aggregation weights n_g/Σn_g are ESTIMATED and the aggregate IF needs
    // the weight-estimation (wif) term. Without it the overall SE was ~25%
    // understated on this DGP (95% CI coverage ~0.86). Seeded → deterministic.
    let n_rep = 400usize;
    let (n, t) = (120usize, 12usize);
    // Population estimand: cells (g=3: 9 post periods, tau=1), (g=6: 6, tau=4),
    // cohort probabilities .3/.3 → theta = (.3*9*1 + .3*6*4)/(.3*9 + .3*6).
    let theta = (0.3 * 9.0 * 1.0 + 0.3 * 6.0 * 4.0) / (0.3 * 9.0 + 0.3 * 6.0);
    let mut covered = 0usize;
    for rep in 0..n_rep {
        let mut rng = Xoshiro256pp::seed_from_u64(40_000 + rep as u64);
        let mut y = Mat::zeros(n, t);
        let mut starts: Vec<Option<usize>> = vec![None; n];
        for i in 0..n {
            let u = rng.next_f64();
            let (g, tau) = if u < 0.3 {
                (Some(3usize), 1.0)
            } else if u < 0.6 {
                (Some(6usize), 4.0)
            } else {
                (None, 0.0)
            };
            starts[i] = g;
            for p in 0..t {
                let eff = match g {
                    Some(gg) if p >= gg => tau,
                    _ => 0.0,
                };
                y.set(i, p, eff + rng.next_normal());
            }
        }
        let panel = Panel::new(y, starts);
        let cs = fit_callaway(&panel);
        if (cs.overall_att - theta).abs() <= 1.96 * cs.overall_se {
            covered += 1;
        }
    }
    let coverage = covered as f64 / n_rep as f64;
    assert!(
        coverage >= 0.91,
        "overall-ATT CI under-covers with estimated weights: {coverage} (want ~0.95)"
    );
}

#[test]
fn cs_covariate_ci_covers_with_estimated_beta() {
    // The covariate-adjusted IF must include the first-step OLS term
    // -(X̄_g - X̄_c)' ψ_β; without it the SEs were ~35% understated on a
    // confounded DGP (coverage ~0.82). Seeded → deterministic.
    let n_rep = 300usize;
    let (n, t, g0) = (100usize, 10usize, 4usize);
    let tau = 1.0;
    let mut covered = 0usize;
    for rep in 0..n_rep {
        let mut rng = Xoshiro256pp::seed_from_u64(70_000 + rep as u64);
        let mut y = Mat::zeros(n, t);
        let mut xv = vec![0.0; n];
        let mut starts: Vec<Option<usize>> = vec![None; n];
        for i in 0..n {
            let x = rng.next_normal();
            xv[i] = x;
            // Confounded adoption: high-x units more likely treated.
            if x + 0.5 * rng.next_normal() > 0.8 {
                starts[i] = Some(g0);
            }
            for p in 0..t {
                let eff = match starts[i] {
                    Some(gg) if p >= gg => tau,
                    _ => 0.0,
                };
                // x drives a differential TREND (what the adjustment removes).
                y.set(i, p, 0.4 * x * p as f64 + eff + rng.next_normal());
            }
        }
        // Reps where (almost) nobody adopts are not estimable — skip rare ones.
        if starts.iter().filter(|s| s.is_some()).count() < 5 {
            continue;
        }
        let xmat = Mat::from_col_vec(&xv);
        let panel = Panel::new(y, starts).with_covariates(xmat);
        let cs = fit_callaway(&panel);
        if (cs.overall_att - tau).abs() <= 1.96 * cs.overall_se {
            covered += 1;
        }
    }
    let coverage = covered as f64 / n_rep as f64;
    assert!(
        coverage >= 0.90,
        "covariate-adjusted CI under-covers: {coverage} (want ~0.95)"
    );
}

#[test]
fn cs_group_aggregation_weights_cohorts_by_size() {
    // Noiseless heterogeneous panel with unequal cohort sizes and unequal
    // post-window lengths. The "simple" overall weights each post (g,t) CELL
    // by cohort size (longer-exposed cohorts count more); the "group" overall
    // weights each COHORT by size — C&S's recommended headline.
    let (n, t) = (30usize, 12usize);
    let (g1, g2) = (3usize, 9usize);
    let (tau1, tau2) = (1.0, 4.0);
    let (n1, n2) = (5usize, 15usize); // 10 never-treated
    let mut y = Mat::zeros(n, t);
    let mut starts = vec![None; n];
    for i in 0..n {
        let (g, tau) = if i < n1 {
            (Some(g1), tau1)
        } else if i < n1 + n2 {
            (Some(g2), tau2)
        } else {
            (None, 0.0)
        };
        starts[i] = g;
        for p in 0..t {
            let mut v = (i as f64) * 0.1; // unit level only
            if let Some(gg) = g {
                if p >= gg {
                    v += tau;
                }
            }
            y.set(i, p, v);
        }
    }
    let cs = fit_callaway(&Panel::new(y, starts));

    // Per-cohort ATTs are exact.
    assert_eq!(cs.group_study.len(), 2);
    for agg in &cs.group_study {
        let want = if agg.key == g1 as i64 { tau1 } else { tau2 };
        assert!(
            (agg.att - want).abs() < 1e-10,
            "ATT_g for g={} is {}, want {}",
            agg.key,
            agg.att,
            want
        );
    }
    // Group overall: (n1·tau1 + n2·tau2) / (n1 + n2).
    let want_group = (n1 as f64 * tau1 + n2 as f64 * tau2) / (n1 + n2) as f64;
    assert!(
        (cs.overall_group_att - want_group).abs() < 1e-10,
        "group overall {} want {}",
        cs.overall_group_att,
        want_group
    );
    // Simple overall: cell-weighted — different whenever exposure lengths differ.
    let cells1 = (t - g1) as f64 * n1 as f64;
    let cells2 = (t - g2) as f64 * n2 as f64;
    let want_simple = (cells1 * tau1 + cells2 * tau2) / (cells1 + cells2);
    assert!(
        (cs.overall_att - want_simple).abs() < 1e-10,
        "simple overall {} want {}",
        cs.overall_att,
        want_simple
    );
    assert!(
        (want_group - want_simple).abs() > 0.1,
        "DGP should separate them"
    );
}

#[test]
fn cs_anticipation_shifts_the_base_period() {
    // Effect turns on ONE period before formal adoption (anticipation = 1).
    // With anticipation=0 the base period g-1 is contaminated and the
    // event-study dynamics are biased; with anticipation=1 the base moves to
    // g-2 and everything is exact: e = -1 shows the anticipation response,
    // e >= 0 the full effect, e < -1 nothing.
    let (n, t, g) = (12usize, 12usize, 6usize);
    let tau = 2.0;
    let mut y = Mat::zeros(n, t);
    let mut starts = vec![None; n];
    for i in 0..n {
        let treated = i < 4;
        if treated {
            starts[i] = Some(g);
        }
        for p in 0..t {
            let mut v = (i as f64) * 0.3;
            if treated && p + 1 >= g {
                v += tau; // anticipation: effect starts at g-1
            }
            y.set(i, p, v);
        }
    }
    let panel = Panel::new(y, starts);

    let cs0 = fit_callaway(&panel);
    let e_minus2_biased = cs0
        .event_study
        .iter()
        .find(|e| e.key == -2)
        .map(|e| e.att)
        .unwrap_or(0.0);
    // Contaminated base (g-1 carries the effect): placebo at e=-2 looks like -tau.
    assert!(
        (e_minus2_biased + tau).abs() < 1e-10,
        "expected contaminated pre-coefficient ~{}, got {}",
        -tau,
        e_minus2_biased
    );

    let cs1 = fit_callaway_with_anticipation(&panel, ControlGroup::NeverTreated, 1);
    for e in &cs1.event_study {
        let want = if e.key >= -1 { tau } else { 0.0 };
        assert!(
            (e.att - want).abs() < 1e-10,
            "anticipation=1: e={} att {} want {}",
            e.key,
            e.att,
            want
        );
    }
    assert!((cs1.overall_att - tau).abs() < 1e-10);
}
