"""Python-layer tests for the synthetic-control estimator."""

import numpy as np
import pytest

from panelkit import SyntheticControl


def planted_panel(tau, seed=0):
    rng = np.random.default_rng(seed)
    N, T, T0 = 8, 30, 22
    y = np.zeros((N, T))
    for u in range(1, N):
        level = rng.normal()
        for t in range(T):
            level += 0.3 * rng.normal()
            y[u, t] = 10.0 + level + 0.5 * u
    for t in range(T):
        base = 0.6 * y[1, t] + 0.4 * y[2, t]
        y[0, t] = base + (tau if t >= T0 else 0.0)
    return y, T0


def test_recovers_planted_effect():
    y, t0 = planted_panel(3.0)
    res = SyntheticControl().fit(y, treated=[0], treat_time=t0)
    assert abs(res.att - 3.0) < 1e-2
    assert res.pre_rmspe < 1e-3
    np.testing.assert_allclose(res.att_path, 3.0, atol=1e-2)


def test_weights_simplex():
    y, t0 = planted_panel(1.0)
    res = SyntheticControl().fit(y, treated=[0], treat_time=t0)
    assert abs(res.weights.sum() - 1.0) < 1e-8
    assert (res.weights >= -1e-9).all()


def test_zero_effect():
    y, t0 = planted_panel(0.0)
    res = SyntheticControl().fit(y, treated=[0], treat_time=t0)
    assert abs(res.att) < 1e-2


def test_placebo_inference_attaches_pvalue():
    y, t0 = planted_panel(3.0)
    res = SyntheticControl(inference="placebo").fit(y, treated=[0], treat_time=t0)
    assert res.p_value is not None
    assert 0.0 < res.p_value <= 1.0
    assert res.inference_distribution is not None


@pytest.mark.parametrize("kind", ["block", "stationary"])
def test_bootstrap_inference_attaches_se_and_ci(kind):
    y, t0 = planted_panel(3.0)
    res = SyntheticControl(inference=kind, n_reps=1000, seed=1).fit(
        y, treated=[0], treat_time=t0
    )
    assert res.se is not None and res.se >= 0.0
    assert res.ci is not None and res.ci[0] <= res.att <= res.ci[1]


def test_bootstrap_deterministic_across_calls():
    y, t0 = planted_panel(3.0)
    a = SyntheticControl(inference="block", seed=7).fit(y, treated=[0], treat_time=t0)
    b = SyntheticControl(inference="block", seed=7).fit(y, treated=[0], treat_time=t0)
    assert a.se == b.se and a.ci == b.ci


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        SyntheticControl().fit(np.zeros(5), treated=[0], treat_time=2)
