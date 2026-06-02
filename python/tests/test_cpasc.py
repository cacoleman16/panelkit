"""Python-layer tests for the CP-ASC family (multiple treated units)."""

import numpy as np
import pytest

from panelkit import CPASC


def multi_treated(n_treated, tau, seed=0):
    rng = np.random.default_rng(seed)
    n_donor = 8
    N = n_treated + n_donor
    T, T0 = 30, 22
    Y = np.zeros((N, T))
    for u in range(n_treated, N):
        level = rng.normal()
        for t in range(T):
            level += 0.3 * rng.normal()
            Y[u, t] = 10.0 + level + 0.5 * u
    d1, d2 = n_treated, n_treated + 1
    treated = list(range(n_treated))
    for u in range(n_treated):
        a = 0.4 + 0.1 * u
        for t in range(T):
            base = a * Y[d1, t] + (1 - a) * Y[d2, t]
            Y[u, t] = base + (tau if t >= T0 else 0.0) + 0.05 * rng.normal()
    return Y, treated, T0


@pytest.mark.parametrize("mode", ["mspe", "stratified", "cumulative"])
def test_modes_recover_pooled_effect(mode):
    Y, treated, t0 = multi_treated(6, 2.0, seed=1)
    res = CPASC(mode=mode).fit(Y, treated, t0)
    assert abs(res.att - 2.0) < 0.6
    assert abs(res.unit_weight.sum() - 1.0) < 1e-9
    assert len(res.unit_ids) == 6


def test_conformal_pvalue_small_under_effect():
    Y, treated, t0 = multi_treated(6, 2.0, seed=2)
    res = CPASC().fit(Y, treated, t0)
    assert res.p_value < 0.2


def test_conformal_pvalue_not_significant_under_null():
    Y, treated, t0 = multi_treated(6, 0.0, seed=3)
    res = CPASC().fit(Y, treated, t0)
    assert res.p_value > 0.05


def test_bad_mode_raises():
    Y, treated, t0 = multi_treated(4, 1.0, seed=4)
    with pytest.raises(ValueError):
        CPASC(mode="nonsense").fit(Y, treated, t0)
