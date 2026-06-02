"""Python-layer tests for the DiD family, including TWFE bias under staggered
heterogeneous adoption."""

import numpy as np
import pytest

from panelkit import TWFE, CallawaySantAnna, GoodmanBacon, SunAbraham


def staggered(eff_early, eff_late, seed=0):
    rng = np.random.default_rng(seed)
    per_group, T = 25, 16
    N = per_group * 3
    g1, g2 = 5, 10
    unit_fe = 3.0 + rng.normal(size=N)
    time_fe = np.cumsum(0.05 * rng.normal(size=T))
    treat_start, Y = [], np.zeros((N, T))
    es = ec = 0.0
    for i in range(N):
        grp = i // per_group
        start, eff = {0: (None, 0.0), 1: (g1, eff_early), 2: (g2, eff_late)}[grp]
        treat_start.append(-1 if start is None else start)
        for t in range(T):
            v = unit_fe[i] + time_fe[t] + 0.1 * rng.normal()
            if start is not None and t >= start:
                v += eff
                es += eff
                ec += 1
            Y[i, t] = v
    return Y, treat_start, es / ec


def test_callaway_recovers_truth():
    Y, ts, true_att = staggered(1.0, 5.0, seed=1)
    res = CallawaySantAnna().fit(Y, ts)
    assert abs(res.att - true_att) < 0.4
    assert res.se > 0
    assert len(res.event_time) == len(res.event_att) == len(res.event_se)


def test_sunab_recovers_truth():
    Y, ts, true_att = staggered(1.0, 5.0, seed=2)
    res = SunAbraham().fit(Y, ts)
    assert abs(res.att - true_att) < 0.4


def test_twfe_more_biased_than_modern():
    Y, ts, true_att = staggered(1.0, 8.0, seed=3)
    twfe = TWFE().fit(Y, ts).att
    cs = CallawaySantAnna().fit(Y, ts).att
    assert abs(twfe - true_att) > abs(cs - true_att)


def test_none_means_never_treated():
    # Accept None as never-treated marker.
    Y, ts, true_att = staggered(2.0, 2.0, seed=4)
    ts2 = [None if c < 0 else c for c in ts]
    res = CallawaySantAnna().fit(Y, ts2)
    assert abs(res.att - true_att) < 0.4


def test_bacon_reproduces_twfe():
    Y, ts, _ = staggered(1.0, 8.0, seed=5)
    twfe = TWFE().fit(Y, ts).att
    bacon = GoodmanBacon().fit(Y, ts)
    assert abs(bacon.twfe - twfe) < 1e-9
    assert sum(c.weight for c in bacon.components) == pytest.approx(1.0)


def test_bacon_has_forbidden_weight():
    Y, ts, _ = staggered(1.0, 8.0, seed=6)
    bacon = GoodmanBacon().fit(Y, ts)
    assert bacon.forbidden_weight > 0.0
    kinds = {c.kind for c in bacon.components}
    assert "later_vs_earlier_forbidden" in kinds
