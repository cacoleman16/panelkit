"""Tests for the geo design layer (power, selection, report, from_long)."""

import numpy as np
import pytest

from panelkit.design import GeoDesign


def geo_panel(n=20, t=60, seed=1):
    rng = np.random.default_rng(seed)
    names = [f"M{i:02d}" for i in range(n)]
    Y = np.zeros((n, t))
    for u in range(1, n):
        lv = 5 + rng.normal()
        for p in range(t):
            lv += 0.2 * rng.normal()
            Y[u, p] = 50 + 10 * lv + 3 * u
    for p in range(t):
        Y[0, p] = 0.5 * Y[1, p] + 0.5 * Y[2, p] + 0.5 * rng.normal()
    return Y, names


def test_power_report_basics():
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    rep = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05, 0.1, 0.2])
    assert set(rep.results) == {"SC", "ASC", "SDID"}
    assert rep.recommended == "SDID"
    # power rises with lift for the recommended method
    pw = rep.best.power
    assert pw[-1] >= pw[0]
    # summary + repr render
    s = rep.summary()
    assert "GEO TEST DESIGN REPORT" in s and "MDE" in s.upper()
    assert 0 <= rep.confidence <= 100


def test_treated_by_name_or_index_equivalent():
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    a = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.1]).best.power
    b = d.power(treated=[0], test_len=10, lifts=[0.0, 0.1]).best.power
    assert a == b


def test_plot_writes_file(tmp_path):
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    rep = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05, 0.1])
    out = tmp_path / "p.png"
    rep.plot(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_market_selection_returns_ranked():
    Y, names = geo_panel(n=14)
    d = GeoDesign(Y, names=names)
    ranked = d.select_markets(test_len=10, target_lift=0.1, max_treated=3,
                              n_candidates=30, top=5)
    assert 1 <= len(ranked) <= 5
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(c["markets"][0], str) for c in ranked)


def test_unknown_market_raises():
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    with pytest.raises(ValueError):
        d.power(treated=["nope"], test_len=10)


def test_from_long_roundtrip():
    pd = pytest.importorskip("pandas")
    Y, names = geo_panel(n=6, t=12)
    rows = []
    for i, nm in enumerate(names):
        for t in range(Y.shape[1]):
            rows.append({"dma": nm, "week": t, "sales": Y[i, t]})
    df = pd.DataFrame(rows)
    d = GeoDesign.from_long(df, location="dma", time="week", outcome="sales")
    assert d.n == 6 and d.t == 12
    np.testing.assert_allclose(np.sort(d.Y, axis=0), np.sort(Y, axis=0), rtol=1e-9)
