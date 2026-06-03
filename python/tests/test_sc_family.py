"""Python-layer tests for ASC, SDID and MC-NNM on a low-rank factor panel."""

import numpy as np
import pytest

from panelkit import AugmentedSC, MCNNM, SyntheticDiD


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
