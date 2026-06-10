"""Placebo SE/CI must be in ATT (outcome) units, and ASC must be
translation-invariant. Both were audit findings: the placebo `se` used to be
the standard deviation of the dimensionless RMSPE *ratios* (so it didn't move
when the outcomes were rescaled ×1000), and ASC's ridge was fitted without an
intercept (so adding a constant to every outcome changed the ATT)."""

import numpy as np
import pytest

from panelkit import AugmentedSC, SyntheticControl


def _panel(n=12, t=30, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 8.0, size=(n, 1))
    trend = np.linspace(0.0, 4.0, t)
    return base + trend + rng.normal(0.0, 1.5, size=(n, t))


def test_placebo_se_and_ci_are_att_scale():
    y = _panel()
    t0 = 22
    res = SyntheticControl(inference="placebo").fit(y, treated=[0], treat_time=t0)
    res_k = SyntheticControl(inference="placebo").fit(y * 1000.0, treated=[0], treat_time=t0)

    # The p-value is scale-free; the SE and CI live in outcome units.
    assert res.p_value == pytest.approx(res_k.p_value, abs=1e-12)
    assert res_k.se == pytest.approx(res.se * 1000.0, rel=1e-9)
    assert res_k.ci[0] == pytest.approx(res.ci[0] * 1000.0, rel=1e-9)
    assert res_k.ci[1] == pytest.approx(res.ci[1] * 1000.0, rel=1e-9)

    # se is exactly the spread of the ATT-scale placebo null...
    assert res.se == pytest.approx(float(np.std(res.placebo_atts, ddof=1)), rel=1e-12)
    # ...not of the dimensionless ratio distribution.
    assert res.se != pytest.approx(float(np.std(res.inference_distribution, ddof=1)), rel=1e-6)

    # CI = att + placebo-ATT null quantiles, and it brackets the estimate
    # whenever the null brackets zero.
    lo, hi = res.ci
    assert lo < hi
    assert len(res.placebo_atts) == len(res.inference_distribution)


def test_placebo_ci_centered_on_att_under_null():
    # No real effect: the CI should cover ~zero and sit around the (small) att.
    y = _panel(seed=11)
    res = SyntheticControl(inference="placebo").fit(y, treated=[0], treat_time=22)
    lo, hi = res.ci
    assert lo <= res.att <= hi


def test_asc_translation_invariance():
    y = _panel(seed=3)
    y[0, 22:] += 5.0  # planted effect
    for kwargs in ({}, {"aug_lambda": 10.0}):
        a0 = AugmentedSC(**kwargs).fit(y, [0], 22).att
        for shift in (100.0, 1e4):
            a1 = AugmentedSC(**kwargs).fit(y + shift, [0], 22).att
            assert a1 == pytest.approx(a0, abs=1e-6), (
                f"ASC changed under +{shift} shift: {a0} -> {a1} ({kwargs})"
            )


def test_asc_still_matches_sc_under_exact_fit():
    # Exact convex replica in the donor pool: imbalance = 0, so ASC == SC and
    # both recover the planted effect exactly.
    rng = np.random.default_rng(0)
    t = 24
    donors = rng.normal(50.0, 5.0, size=(4, t))
    treated = 0.25 * donors.sum(axis=0)
    y = np.vstack([treated, donors])
    tau = 3.0
    y[0, 16:] += tau
    att = AugmentedSC().fit(y, [0], 16).att
    assert att == pytest.approx(tau, abs=1e-8)
