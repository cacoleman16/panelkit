//! DiD family correctness, including the headline modern-DiD result: under
//! staggered adoption with heterogeneous effects, TWFE is biased while
//! Callaway–Sant'Anna and Sun–Abraham recover the truth.

#![allow(clippy::needless_range_loop)]

use panelkit_estimators::did::{
    bacon_decompose, fit_callaway, fit_callaway_with, fit_sunab, fit_twfe, BaconKind, ControlGroup,
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

    let simple = fit_callaway_with(&panel, ControlGroup::NeverTreated).overall_att;
    // covariate-adjusted: covariates are attached, so fit uses regression adjustment
    let adjusted = fit_callaway_with(&panel, ControlGroup::NeverTreated).overall_att;
    // NOTE: fit_callaway_with auto-uses covariates when present, so to compare we
    // fit the simple version on a covariate-free copy.
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

    let _ = simple;
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
