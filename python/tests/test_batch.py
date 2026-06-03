"""Tests for batched/parallel fitting (Monte-Carlo / power-analysis path)."""

import numpy as np
import pytest

from panelkit import AugmentedSC, SyntheticControl, SyntheticDiD


def make_stack(R, tau, seed=0, N=20, T=30, T0=22):
    rng = np.random.default_rng(seed)
    stack = np.zeros((R, N, T))
    for r in range(R):
        for u in range(1, N):
            lv = rng.normal()
            for t in range(T):
                lv += 0.3 * rng.normal()
                stack[r, u, t] = 10 + lv + u
        for t in range(T):
            stack[r, 0, t] = (0.6 * stack[r, 1, t] + 0.4 * stack[r, 2, t]
                              + (tau if t >= T0 else 0.0) + 0.05 * rng.normal())
    return stack, T0


@pytest.mark.parametrize("model", [SyntheticControl(), AugmentedSC(), SyntheticDiD()])
def test_fit_many_shape_and_recovery(model):
    stack, t0 = make_stack(200, 2.0, seed=1)
    atts = model.fit_many(stack, treated=[0], treat_time=t0)
    assert atts.shape == (200,)
    # Mean ATT across reps should be near the planted effect.
    assert abs(atts.mean() - 2.0) < 0.2


def test_fit_many_matches_single_fit():
    stack, t0 = make_stack(10, 1.5, seed=2)
    model = SyntheticControl()
    batch = model.fit_many(stack, treated=[0], treat_time=t0)
    for r in range(stack.shape[0]):
        single = model.fit(stack[r], treated=[0], treat_time=t0).att
        assert abs(batch[r] - single) < 1e-9


def test_fit_many_bad_shape_raises():
    with pytest.raises(ValueError):
        SyntheticControl().fit_many(np.zeros((5, 5)), treated=[0], treat_time=2)
