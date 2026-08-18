"""Python-layer tests for the SC family on a low-rank factor panel: ASC, SDID,
MC-NNM, the Ferman-Pinto demeaned estimator, robust SC, and donor weight bounds."""

import numpy as np
import pytest

from panelkit import (
    AugmentedSC,
    DemeanedSC,
    MCNNM,
    RobustSC,
    SyntheticControl,
    SyntheticDiD,
)


def factor_panel(tau, seed=0, n=20, t=30, t0=24):
    rng = np.random.default_rng(seed)
    uf = rng.normal(size=(n, 2))
    tf = 0.5 * rng.normal(size=(t, 2))
    ul = 10.0 + rng.normal(size=n)
    tl = np.cumsum(0.02 * rng.normal(size=t))
    y = ul[:, None] + tl[None, :] + uf @ tf.T
    y[0, t0:] += tau
    return y, t0


@pytest.mark.parametrize("model_cls", [AugmentedSC, SyntheticDiD, MCNNM])
def test_recovers_effect(model_cls):
    y, t0 = factor_panel(2.0, seed=1)
    res = model_cls().fit(y, treated=[0], treat_time=t0)
    assert abs(res.att - 2.0) < 0.8


@pytest.mark.parametrize("model_cls", [AugmentedSC, SyntheticDiD, MCNNM])
def test_zero_effect(model_cls):
    y, t0 = factor_panel(0.0, seed=2)
    res = model_cls().fit(y, treated=[0], treat_time=t0)
    assert abs(res.att) < 0.8


def test_sdid_weights_simplex():
    y, t0 = factor_panel(2.0, seed=3)
    res = SyntheticDiD().fit(y, treated=[0], treat_time=t0)
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert (res.weights >= -1e-9).all()


def test_mcnnm_deterministic():
    y, t0 = factor_panel(3.0, seed=4)
    a = MCNNM(seed=123).fit(y, treated=[0], treat_time=t0).att
    b = MCNNM(seed=123).fit(y, treated=[0], treat_time=t0).att
    assert a == b  # same seed -> identical CV hold-out -> identical result


def test_mcnnm_truncated_svd_recovers_effect():
    # max_rank switches to the fast randomized truncated SVD; on a low-rank
    # panel it should land near the full-SVD answer.
    y, t0 = factor_panel(3.0, seed=9)
    full = MCNNM(lambda_=1.0).fit(y, treated=[0], treat_time=t0).att
    fast = MCNNM(lambda_=1.0, max_rank=6).fit(y, treated=[0], treat_time=t0).att
    assert abs(fast - full) < 0.5


def test_fit_mcnnm_lambda_keyword_usable():
    # Regression: the raw binding's penalty arg must be usable as a Python
    # keyword (`lambda_`, not the reserved word `lambda`).
    from panelkit import _panelkit
    y, t0 = factor_panel(3.0, seed=2)
    r = _panelkit.fit_mcnnm(y, [0], int(t0), lambda_=1.0)
    assert r.att == r.att  # not NaN; call succeeded with the keyword


# ---------------------------------------------------------------------------
# Demeaned SC (Ferman-Pinto), robust SC, and the donor weight bounds.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [DemeanedSC, RobustSC])
def test_new_members_recover_effect(model_cls):
    y, t0 = factor_panel(2.0, seed=11)
    res = model_cls().fit(y, treated=[0], treat_time=t0)
    assert abs(res.att - 2.0) < 0.8


def test_demeaned_sc_handles_a_level_gap_that_breaks_plain_sc():
    y, t0 = factor_panel(2.0, seed=12)
    y = y.copy()
    y[0] += 50.0                      # treated level far outside the donor hull
    fp = DemeanedSC().fit(y, treated=[0], treat_time=t0)
    sc = SyntheticControl().fit(y, treated=[0], treat_time=t0)
    assert abs(fp.att - 2.0) < 0.8
    assert abs(sc.att - 2.0) > 5 * abs(fp.att - 2.0)
    assert abs(fp.weights.sum() - 1.0) < 1e-6 and (fp.weights >= -1e-9).all()


def test_demeaned_sc_ignores_a_treated_level_shift():
    y, t0 = factor_panel(1.0, seed=13)
    base = DemeanedSC().fit(y, treated=[0], treat_time=t0).att
    shifted = DemeanedSC().fit(y + 0.0 * y, treated=[0], treat_time=t0).att
    y2 = y.copy()
    y2[0] += 987.65
    moved = DemeanedSC().fit(y2, treated=[0], treat_time=t0).att
    assert abs(base - shifted) < 1e-9
    assert abs(base - moved) < 1e-8


def test_robust_sc_weights_are_unconstrained():
    y, t0 = factor_panel(2.0, seed=14)
    res = RobustSC().fit(y, treated=[0], treat_time=t0)
    # Not a convex combination — that freedom is the estimator's whole point.
    assert res.weights.size == y.shape[0] - 1
    assert np.isfinite(res.weights).all()


@pytest.mark.parametrize("model_cls", [SyntheticControl, AugmentedSC, SyntheticDiD,
                                       DemeanedSC])
def test_max_weight_caps_donor_concentration(model_cls):
    y, t0 = factor_panel(2.0, seed=15, n=25, t=40, t0=32)
    free = model_cls().fit(y, treated=[0], treat_time=t0)
    capped = model_cls(max_weight=0.12).fit(y, treated=[0], treat_time=t0)
    assert free.weights.max() > 0.12, "panel does not exercise the cap"
    assert capped.weights.max() <= 0.12 + 1e-7
    assert abs(capped.weights.sum() - 1.0) < 1e-6


@pytest.mark.parametrize("model_cls", [SyntheticControl, AugmentedSC, SyntheticDiD,
                                       DemeanedSC])
def test_min_weight_forces_diversification(model_cls):
    y, t0 = factor_panel(2.0, seed=16, n=25, t=40, t0=32)
    floored = model_cls(min_weight=0.02).fit(y, treated=[0], treat_time=t0)
    assert floored.weights.min() >= 0.02 - 1e-7
    assert abs(floored.weights.sum() - 1.0) < 1e-6


def test_weight_bounds_are_validated():
    y, t0 = factor_panel(2.0, seed=17, n=10)
    with pytest.raises(ValueError, match="max_weight"):
        SyntheticControl(max_weight=1.5)
    with pytest.raises(ValueError, match="min_weight"):
        SyntheticControl(min_weight=-0.1)
    with pytest.raises(ValueError, match="must not exceed"):
        SyntheticControl(min_weight=0.5, max_weight=0.2)
    # Feasibility depends on the donor count, so it is raised at fit time.
    with pytest.raises(ValueError, match="infeasible"):
        SyntheticControl(max_weight=0.05).fit(y, treated=[0], treat_time=t0)
    with pytest.raises(ValueError, match="infeasible"):
        SyntheticControl(min_weight=0.5).fit(y, treated=[0], treat_time=t0)


def test_penalized_sc_stays_on_the_simplex_and_changes_the_weights():
    y, t0 = factor_panel(2.0, seed=18, n=25, t=40, t0=32)
    plain = SyntheticControl().fit(y, treated=[0], treat_time=t0)
    pen = SyntheticControl(penalty=0.5).fit(y, treated=[0], treat_time=t0)
    assert abs(pen.weights.sum() - 1.0) < 1e-6 and (pen.weights >= -1e-9).all()
    assert not np.allclose(plain.weights, pen.weights)


def test_new_members_fit_many_matches_single_fits():
    y, t0 = factor_panel(2.0, seed=19)
    stack = np.stack([y, y * 1.01])
    for cls, kw in [(DemeanedSC, {}), (RobustSC, {})]:
        atts = cls(**kw).fit_many(stack, treated=[0], treat_time=t0)
        one = cls(**kw).fit(y, treated=[0], treat_time=t0).att
        assert atts.shape == (2,)
        assert abs(atts[0] - one) < 1e-9


def test_placebo_inference_works_for_new_members():
    y, t0 = factor_panel(3.0, seed=20, n=25, t=40, t0=32)
    for cls in (DemeanedSC, RobustSC):
        res = cls(inference="placebo").fit(y, treated=[0], treat_time=t0)
        assert res.p_value is not None and 0.0 < res.p_value <= 1.0
        assert res.ci is not None
