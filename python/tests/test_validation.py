"""Every input in this file used to abort the host process (SIGABRT through
panic="abort"), silently misbehave, or raise an unhelpful panic. They must all
either raise a clean ValueError or behave sensibly now."""

import numpy as np
import pytest

from panelkit import (
    _panelkit,
    AugmentedSC,
    CallawaySantAnna,
    CPASC,
    GoodmanBacon,
    MCNNM,
    SunAbraham,
    SyntheticControl,
    SyntheticDiD,
)
from panelkit.design import GeoDesign


def _panel(n=10, t=30, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 5.0, size=(n, 1))
    return base + rng.normal(0.0, 2.0, size=(n, t))


# --- geo design layer ------------------------------------------------------


def test_power_test_len_out_of_range():
    d = GeoDesign(_panel())
    for bad in (0, 30, 31, -1):
        with pytest.raises(ValueError, match="test_len"):
            d.power(treated=[0], test_len=bad)


def test_power_test_len_leaves_no_pre_window():
    # 16-period test on a 30-period panel: max(16,2) > 30-16 → must be a clean
    # error (used to abort the interpreter at power.rs:141).
    d = GeoDesign(_panel())
    with pytest.raises(ValueError, match="Max usable test_len"):
        d.power(treated=[0], test_len=16)


def test_diagnose_test_len_out_of_range():
    d = GeoDesign(_panel(n=6, t=20))
    with pytest.raises(ValueError, match="test_len"):
        d.diagnose(treated=[0], test_len=25)   # used to abort: capacity overflow
    with pytest.raises(ValueError, match="test_len"):
        d.diagnose(treated=[0], test_len=20)   # zero pre-periods → vacuous design


def test_all_markets_treated_is_an_error():
    d = GeoDesign(_panel(n=5))
    with pytest.raises(ValueError, match="control"):
        d.evaluate(treated=[0, 1, 2, 3, 4], treat_start=20)
    with pytest.raises(ValueError, match="control"):
        d.power(treated=[0, 1, 2, 3, 4], test_len=5)
    with pytest.raises(ValueError, match="control"):
        d.diagnose(treated=[0, 1, 2, 3, 4], test_len=5)


def test_negative_lifts_rejected():
    d = GeoDesign(_panel())
    with pytest.raises(ValueError, match="lifts"):
        d.power(treated=[0], test_len=5, lifts=[-0.05, 0.05])


def test_zero_baseline_panel_does_not_crash():
    # All-zero outcomes: est-% of baseline is undefined (NaN) but the call must
    # survive (used to panic in a sort over NaNs and abort the process).
    res = _panelkit.geo_power(
        np.zeros((6, 30)), [0], 5, [0.0, 0.05], "sc", 0.1, 0.8, 5, None
    )
    assert all(np.isfinite(p) for p in res.power)


def test_select_markets_exact_size_vs_include():
    d = GeoDesign(_panel())
    with pytest.raises(ValueError, match="exact_size"):
        d.select_markets(test_len=5, target_lift=0.05, max_treated=3,
                         exact_size=2, include=[0, 1, 2])


def test_select_markets_duplicate_eligible_dedup():
    d = GeoDesign(_panel())
    ranked = d.select_markets(test_len=5, target_lift=0.05, max_treated=2,
                              eligible=[3, 3, 4, 5], exact_size=2, n_candidates=20)
    for cand in ranked:
        assert len(cand["markets"]) == len(set(cand["markets"])) == 2


def test_recommended_is_case_insensitive_and_validated():
    d = GeoDesign(_panel())
    rep = d.power(treated=[0], test_len=5, recommended="ensemble")
    assert rep.recommended == "ENSEMBLE"
    with pytest.raises(ValueError, match="recommended"):
        d.power(treated=[0], test_len=5, recommended="bogus")


def test_methods_validated():
    d = GeoDesign(_panel())
    with pytest.raises(ValueError, match="methods"):
        d.power(treated=[0], test_len=5, methods=[], ensemble=False)
    with pytest.raises(ValueError, match="unknown methods"):
        d.evaluate(treated=[0], treat_start=25, methods=["nope"])


def test_bool_market_spec_rejected():
    d = GeoDesign(_panel())
    with pytest.raises(ValueError, match="bool"):
        d.power(treated=[True], test_len=5)


def test_duplicate_market_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        GeoDesign(_panel(n=4), names=["a", "a", "b", "c"])


def test_alpha_validated():
    d = GeoDesign(_panel())
    for bad in (0.0, 1.0, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            d.power(treated=[0], test_len=5, alpha=bad)


# --- SC family -------------------------------------------------------------


def test_raw_layer_validates_inputs():
    y = _panel()
    with pytest.raises(ValueError, match="out of range"):
        _panelkit.fit_sc(y, [99], 10, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="at least one treated"):
        _panelkit.fit_sc(y, [], 10, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="never-treated"):
        _panelkit.fit_sc(y, list(range(10)), 10, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="non-empty"):
        _panelkit.fit_sc(np.zeros((0, 10)), [0], 5, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="treat_time"):
        _panelkit.fit_sc(y, [0], 0, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="treat_time"):
        _panelkit.fit_sc(y, [0], 30, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="level"):
        _panelkit.fit_sc(y, [0], 10, 0.0, True, 1.0)
    with pytest.raises(ValueError, match="NaN"):
        _panelkit.fit_sc(np.full((5, 10), np.nan), [0], 5, 0.0, False, 0.95)
    with pytest.raises(ValueError, match="more than once"):
        _panelkit.fit_sc(y, [0, 0], 10, 0.0, False, 0.95)


def test_fit_many_validates():
    stack = np.stack([_panel(seed=s) for s in range(3)])
    with pytest.raises(ValueError, match="out of range"):
        _panelkit.fit_many(stack, [50], 10, "sc", 0.0, 1.0)
    bad = stack.copy()
    bad[1, 2, 3] = np.nan
    with pytest.raises(ValueError, match="NaN|non-finite"):
        SyntheticControl().fit_many(bad, [0], 10)


def test_aug_lambda_zero_rejected():
    # λ=0 made the augmentation Gram singular → Cholesky .expect() → process
    # abort. Now a constructor-time error (and a raw-layer error).
    with pytest.raises(ValueError, match="aug_lambda"):
        AugmentedSC(aug_lambda=0.0)
    with pytest.raises(ValueError, match="aug_lambda"):
        _panelkit.fit_asc(_panel(), [0], 10, 0.0, 0.0)
    with pytest.raises(ValueError, match="aug_lambda"):
        CPASC(aug_lambda=0.0)


def test_mcnnm_lambda_zero_rejected():
    with pytest.raises(ValueError, match="lambda_"):
        MCNNM(lambda_=0.0)
    with pytest.raises(ValueError, match="lambda_"):
        _panelkit.fit_mcnnm(_panel(), [0], 10, 0.0, 100, 1e-4, 0, None)


def test_unknown_inference_rejected():
    with pytest.raises(ValueError, match="inference"):
        SyntheticControl(inference="placbo")  # typo must not silently no-op
    with pytest.raises(ValueError, match="inference"):
        AugmentedSC(inference="placbo")
    with pytest.raises(ValueError, match="inference"):
        SyntheticDiD(inference="bootstrap")


def test_level_validated():
    with pytest.raises(ValueError, match="level"):
        SyntheticControl(level=1.0)


def test_non_integral_indices_rejected():
    y = _panel()
    with pytest.raises(ValueError, match="integer"):
        SyntheticControl().fit(y, [0.9], 10)
    with pytest.raises(ValueError, match="integer"):
        SyntheticControl().fit(y, [0], 10.7)
    with pytest.raises(ValueError, match="bool"):
        SyntheticControl().fit(y, [True], 10)


def test_placebo_with_one_donor_reports_no_p_value():
    # 2-unit panel: the only placebo has an empty donor pool → no null
    # distribution. Must be None, not a fake p = 1.0.
    y = _panel(n=2)
    res = SyntheticControl(inference="placebo").fit(y, [0], 20)
    assert res.p_value is None
    assert res.se is None


def test_bootstrap_mean_validates():
    with pytest.raises(ValueError, match="non-empty"):
        _panelkit.bootstrap_mean([], "block", 4, 100, 0, 0.95)
    with pytest.raises(ValueError, match="level"):
        _panelkit.bootstrap_mean([1.0, 2.0], "block", 4, 100, 0, 1.0)


# --- DiD family ------------------------------------------------------------


def test_cohort_beyond_sample_is_never_treated():
    # The R `did` convention: g >= T means "never treated within the sample".
    # Used to abort with an out-of-bounds panic (C&S) / usize underflow (Bacon).
    y = _panel(n=20, t=8, seed=3)
    y[10:15, 4:] += 5.0
    ts_explicit = [-1] * 10 + [4] * 5 + [-1] * 5
    ts_oob = [-1] * 10 + [4] * 5 + [11] * 5
    for cls in (CallawaySantAnna, SunAbraham, GoodmanBacon):
        a = cls().fit(y, ts_explicit)
        b = cls().fit(y, ts_oob)
        if hasattr(a, "att"):
            assert a.att == pytest.approx(b.att, abs=1e-12)
        else:
            assert a.twfe == pytest.approx(b.twfe, abs=1e-12)


def test_callaway_without_never_treated_is_clean_error():
    y = _panel(n=10, t=20, seed=4)
    ts = [5] * 5 + [10] * 5  # every unit eventually adopts
    with pytest.raises(ValueError, match="notyet"):
        CallawaySantAnna().fit(y, ts)
    # ... and the suggested fix works:
    res = CallawaySantAnna(control_group="notyet").fit(y, ts)
    assert np.isfinite(res.att)


def test_sunab_without_never_treated_is_clean_error():
    y = _panel(n=10, t=20, seed=5)
    with pytest.raises(ValueError, match="never-treated"):
        SunAbraham().fit(y, [5] * 5 + [10] * 5)


def test_control_group_validated():
    with pytest.raises(ValueError, match="control_group"):
        CallawaySantAnna(control_group="nver")


def test_treat_start_bool_and_float_rejected():
    y = _panel(n=4, t=10)
    with pytest.raises(ValueError, match="bool"):
        CallawaySantAnna().fit(y, [True, -1, -1, -1])
    with pytest.raises(ValueError, match="integer"):
        CallawaySantAnna().fit(y, [3.5, -1, -1, -1])


# --- DiD degenerate designs (added with the DiD-correctness fixes) ----------


def test_twfe_unidentified_designs_error():
    y = _panel(n=6, t=10)
    with pytest.raises(ValueError, match="no estimable cohort"):
        _panelkit.fit_twfe_py(y, [-1] * 6)          # nobody treated
    with pytest.raises(ValueError, match="no estimable cohort"):
        _panelkit.fit_twfe_py(y, [0] * 6)           # everyone always-treated
    with pytest.raises(ValueError, match="absorb"):
        _panelkit.fit_twfe_py(y, [4] * 6)           # one shared date, no controls


def test_bacon_rejects_always_treated():
    y = _panel(n=8, t=10)
    with pytest.raises(ValueError, match="period 0"):
        GoodmanBacon().fit(y, [0, 0, 4, 4, 7, 7, -1, -1])


def test_cs_and_sa_reject_no_estimable_cohort():
    y = _panel(n=4, t=10)
    with pytest.raises(ValueError, match="no estimable cohort"):
        CallawaySantAnna(control_group="notyet").fit(y, [-1, -1, -1, -1])
    with pytest.raises(ValueError, match="no estimable cohort"):
        SunAbraham().fit(y, [0, 0, -1, -1])
