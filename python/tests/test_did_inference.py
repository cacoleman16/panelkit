"""C&S inference completion: simultaneous bands, anticipation, group
aggregation, and the event-study plot."""

import numpy as np
import pytest

from panelkit import CallawaySantAnna


def _staggered(seed=0, n=60, t=14, tau1=1.0, tau2=4.0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, (n, t))
    ts = [-1] * n
    for i in range(0, 15):
        ts[i] = 4
        y[i, 4:] += tau1
    for i in range(15, 30):
        ts[i] = 9
        y[i, 9:] += tau2
    return y, ts


def test_bootstrap_bands_attach_and_contain_pointwise():
    y, ts = _staggered()
    res = CallawaySantAnna(inference="bootstrap", seed=1).fit(y, ts)
    lo, hi = res.event_bands
    assert len(lo) == len(res.event_time) == len(hi)
    assert res.band_crit >= 1.9          # sup-t >= pointwise z
    # The simultaneous band contains the pointwise 95% interval everywhere.
    assert np.all(lo <= res.event_att - 1.95 * res.event_se + 1e-12)
    assert np.all(hi >= res.event_att + 1.95 * res.event_se - 1e-12)
    # Deterministic given the seed.
    res2 = CallawaySantAnna(inference="bootstrap", seed=1).fit(y, ts)
    assert np.allclose(res.event_bands[0], res2.event_bands[0])


def test_analytic_mode_has_no_bands():
    y, ts = _staggered()
    res = CallawaySantAnna().fit(y, ts)
    assert res.event_bands is None
    assert res.band_crit is None


def test_group_aggregation_exposed():
    y, ts = _staggered(tau1=1.0, tau2=4.0)
    res = CallawaySantAnna().fit(y, ts)
    assert set(res.group_cohort.tolist()) == {4, 9}
    assert res.overall_group_att is not None
    # Equal cohort sizes here -> group overall ~ (1+4)/2; the simple overall
    # overweights the longer-exposed early cohort, so it differs.
    assert res.overall_group_att == pytest.approx(2.5, abs=0.4)
    assert res.overall_group_se > 0.0


def test_anticipation_kwarg():
    # Effect begins one period before formal adoption.
    rng = np.random.default_rng(3)
    n, t, g = 40, 12, 6
    y = rng.normal(0.0, 0.05, (n, t))
    ts = [-1] * n
    for i in range(12):
        ts[i] = g
        y[i, g - 1:] += 2.0
    res0 = CallawaySantAnna().fit(y, ts)
    res1 = CallawaySantAnna(anticipation=1).fit(y, ts)
    # Without anticipation the base period is contaminated -> e=-2 placebo
    # pulled to ~-2; with it, pre-coefficients are clean and the ATT is exact.
    e0 = dict(zip(res0.event_time.tolist(), res0.event_att))
    e1 = dict(zip(res1.event_time.tolist(), res1.event_att))
    assert e0[-2] == pytest.approx(-2.0, abs=0.1)
    # With anticipation=1 the base moves to g-2 (so e=-2 is the base and is
    # omitted); earlier placebos are clean and e=-1 measures the anticipation.
    assert -2 not in e1
    assert abs(e1[-3]) < 0.1
    assert e1[-1] == pytest.approx(2.0, abs=0.1)
    assert res1.att == pytest.approx(2.0, abs=0.05)
    with pytest.raises(ValueError, match="anticipation"):
        CallawaySantAnna(anticipation=-1)


def test_event_study_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    y, ts = _staggered()
    res = CallawaySantAnna(inference="bootstrap").fit(y, ts)
    out = tmp_path / "event.png"
    res.plot(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_cs_inference_kwarg_validated():
    with pytest.raises(ValueError, match="inference"):
        CallawaySantAnna(inference="bootstrp")
