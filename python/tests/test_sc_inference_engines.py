"""Inference engines for the SC family: placebo for ASC/SDID and the
fixed-weights jackknife for SDID — the engines the docs always promised."""

import numpy as np
import pytest

from panelkit import AugmentedSC, SyntheticDiD


def _panel(n=14, t=30, seed=5, tau=0.0, n_treated=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 6.0, size=(n, 1))
    trend = np.linspace(0.0, 3.0, t)
    y = base + trend + rng.normal(0.0, 1.0, size=(n, t))
    y[:n_treated, 22:] += tau
    return y


def test_asc_placebo_attaches_full_inference():
    y = _panel(tau=8.0)
    res = AugmentedSC(inference="placebo").fit(y, [0], 22)
    assert res.p_value is not None and 0.0 < res.p_value <= 1.0
    assert res.p_value <= 0.2          # 8-sigma effect, 13 donors -> rank 1
    assert res.se is not None and res.se > 0.0
    lo, hi = res.ci
    assert lo < hi
    assert len(res.placebo_atts) == len(res.inference_distribution) == 13


def test_sdid_placebo_attaches_full_inference():
    y = _panel(tau=8.0)
    res = SyntheticDiD(inference="placebo").fit(y, [0], 22)
    assert res.p_value is not None and res.p_value <= 0.2
    assert res.se is not None and res.se > 0.0
    lo, hi = res.ci
    assert lo < res.att < hi


def test_sdid_jackknife_se_and_ci():
    y = _panel(tau=3.0, n_treated=3)
    res = SyntheticDiD(inference="jackknife").fit(y, [0, 1, 2], 22)
    assert res.se is not None and res.se > 0.0
    lo, hi = res.ci
    assert lo < res.att < hi
    # The LOO distribution is exposed for inspection.
    assert len(res.inference_distribution) >= 3
    # On this clean DGP the CI should comfortably cover the true effect.
    assert lo < 3.0 < hi


def test_sdid_jackknife_needs_two_treated():
    y = _panel()
    with pytest.raises(ValueError, match="jackknife"):
        SyntheticDiD(inference="jackknife").fit(y, [0], 22)


def test_sdid_jackknife_tight_on_noiseless_additive_panel():
    # Pure unit+time structure: every LOO estimate is (nearly) identical, so
    # the jackknife SE must be ~0.
    n, t, t0 = 10, 20, 14
    rng = np.random.default_rng(0)
    a = rng.normal(50, 5, (n, 1))
    b = rng.normal(0, 2, (1, t))
    y = a + b
    y[:2, t0:] += 2.0
    res = SyntheticDiD(inference="jackknife").fit(y, [0, 1], t0)
    assert res.att == pytest.approx(2.0, abs=1e-8)
    assert res.se == pytest.approx(0.0, abs=1e-8)
