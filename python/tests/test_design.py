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
    assert {"SC", "ASC", "SDID"} <= set(rep.results)
    assert rep.recommended == "SDID"
    # power rises with lift for the recommended method
    pw = rep.best.power
    assert pw[-1] >= pw[0]
    # summary + repr render
    s = rep.summary()
    assert "GEO TEST DESIGN REPORT" in s and "MDE" in s.upper()
    assert 0 <= rep.confidence <= 100


def test_lookback_limits_windows():
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    full = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05])
    recent = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05], lookback=8)
    assert recent.best.n_windows == 8
    assert full.best.n_windows > recent.best.n_windows


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


def test_diagnose_report_and_plot(tmp_path):
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    guard = d.diagnose(treated=["M00"], test_len=10)
    s = guard.summary()
    assert "GUARDRAILS" in s
    assert 0 <= guard.confidence <= 100
    assert 0 < guard.holdout_pct < 1
    out = tmp_path / "g.png"
    guard.plot(str(out))
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


def test_multi_cell_basic_and_disjoint(tmp_path):
    Y, names = geo_panel(n=18, t=60)
    d = GeoDesign(Y, names=names)
    mc = d.multi_cell(
        cells={"A": ["M00", "M01"], "B": ["M02", "M03"], "C": ["M04"]},
        test_len=8, alpha=0.10,
    )
    assert set(mc.cells) == {"A", "B", "C"}
    # each cell measured against the shared pool, donors exclude all treated
    assert len(mc.donor_names) == 18 - 5
    assert all(nm not in {"M00", "M01", "M02", "M03", "M04"} for nm in mc.donor_names)
    # the other cells' markets are NOT in this cell's donor pool / treated set
    assert mc.cells["A"].treated_names == ["M00", "M01"]
    assert 0 < mc.pooled_holdout < 1
    s = mc.summary()
    assert "MULTI-CELL" in s and "A" in s and "B" in s
    out = tmp_path / "mc.png"
    mc.plot(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_multi_cell_rejects_overlap_and_empty_donors():
    Y, names = geo_panel(n=6, t=40)
    d = GeoDesign(Y, names=names)
    # overlapping cells
    with pytest.raises(ValueError, match="disjoint"):
        d.multi_cell(cells={"A": ["M00", "M01"], "B": ["M01"]}, test_len=6)
    # donors overlap a cell
    with pytest.raises(ValueError, match="overlaps"):
        d.multi_cell(cells={"A": ["M00"]}, test_len=6, shared_donors=["M00", "M01"])
    # no markets left for the control pool
    with pytest.raises(ValueError, match="donor"):
        d.multi_cell(cells={"A": [f"M0{i}" for i in range(6)]}, test_len=6)


def test_power_ensemble_present_and_weighted():
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    rep = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05, 0.1])
    assert "ENSEMBLE" in rep.results
    ens = rep.results["ENSEMBLE"]
    w = ens.ensemble_weights
    assert w is not None and len(w) == 3
    assert abs(sum(w) - 1.0) < 1e-9 and all(x >= 0 for x in w)
    # equal weights honored
    eq = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.1],
                 ensemble_weights="equal").results["ENSEMBLE"].ensemble_weights
    assert all(abs(x - 1 / 3) < 1e-9 for x in eq)
    # dict weights honored (normalized)
    dd = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.1],
                 ensemble_weights={"SC": 2, "ASC": 1, "SDID": 1}).results["ENSEMBLE"].ensemble_weights
    assert abs(dd[0] - 0.5) < 1e-9
    # can be turned off
    off = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.1], ensemble=False)
    assert "ENSEMBLE" not in off.results


def test_evaluate_recovers_injected_lift(tmp_path):
    # A realistic donor pool (~40 markets) so in-space placebo inference has power.
    Y, names = geo_panel(n=40, t=72)
    Yt = Y.copy()
    Yt[0, 60:] *= 1.10                       # clear +10% lift on a well-fit market
    d = GeoDesign(Yt, names=names)
    ev = d.evaluate(treated=["M00"], treat_start=60, level=0.90)
    assert set(ev.per) == {"SC", "ASC", "SDID"}
    # ensemble lift recovered in the right ballpark and detected as significant
    assert 0.05 < ev.lift < 0.15
    assert ev.significant
    assert ev.ensemble["n_placebo"] > 0
    assert abs(sum(ev.ensemble["weights"].values()) - 1.0) < 1e-9
    s = ev.summary()
    assert "GEO TEST EVALUATION" in s and "ENSEMBLE" in s
    out = tmp_path / "eval.png"
    ev.plot(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_evaluate_cis_are_calibrated_not_anticonservative():
    # On NULL data (no real effect), a 90% interval must NOT routinely exclude
    # zero. In-space placebo inference should be conservative (FP well under 0.10),
    # never the 50%-ish false-positive rate of a naive post-period bootstrap.
    nm = [f"M{i:02d}" for i in range(40)]
    fp = 0
    K = 20
    for s in range(K):
        Y, _ = geo_panel(n=40, t=60, seed=3000 + s)
        ev = GeoDesign(Y, names=nm).evaluate(treated=["M00"], treat_start=48)
        if ev.significant:
            fp += 1
    assert fp / K <= 0.20      # generous bound; true rate is near 0


def test_evaluate_low_power_is_not_falsely_significant():
    # With only one donor, in-space placebo inference is undefined. It must NOT
    # report a (false) significant effect or a fake zero-width CI.
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(size=40))
    Y = np.vstack([base + rng.normal(scale=0.3, size=40), base])  # market0 ≈ donor
    Y[0, 30:] *= 1.15                                              # big injected lift
    ev = GeoDesign(Y, names=["t", "donor"]).evaluate(treated=["t"], treat_start=30)
    assert ev.ensemble["n_placebo"] <= 1
    assert ev.significant is False
    lo, hi = ev.ensemble["att_lo"], ev.ensemble["att_hi"]
    assert not (np.isfinite(lo) and np.isfinite(hi))   # CI undefined, not zero-width
    assert ev.p_value is None
    assert ev.ensemble["low_power"]


def test_evaluate_dedups_treated_markets():
    # Listing a market twice must not inflate the cumulative (which scales by the
    # number of treated markets).
    Y, names = geo_panel(n=30, t=60)
    Yt = Y.copy()
    Yt[0, 48:] *= 1.10
    d = GeoDesign(Yt, names=names)
    a = d.evaluate(treated=["M00"], treat_start=48)
    b = d.evaluate(treated=["M00", "M00"], treat_start=48)
    assert a.cumulative == b.cumulative
    assert a.lift == b.lift


def test_evaluate_bootstrap_inference_option():
    Y, names = geo_panel(n=30, t=60)
    Yt = Y.copy()
    Yt[0, 48:] *= 1.10
    d = GeoDesign(Yt, names=names)
    ev = d.evaluate(treated=["M00"], treat_start=48, inference="bootstrap")
    e = ev.ensemble
    assert e["inference"] == "block bootstrap"
    assert e["optimistic"] is True
    assert np.isfinite(e["att_lo"]) and np.isfinite(e["att_hi"])
    assert "optimistic" in ev.summary().lower()
    # bootstrap is a fallback where placebo is undefined (1 donor)
    base = 100 + np.cumsum(np.random.default_rng(0).normal(size=40))
    Y1 = np.vstack([base + 0.3 * np.random.default_rng(1).normal(size=40), base])
    Y1[0, 30:] *= 1.15
    g = GeoDesign(Y1, names=["t", "d"])
    assert not np.isfinite(g.evaluate(treated=["t"], treat_start=30).ensemble["att_lo"])
    assert np.isfinite(g.evaluate(treated=["t"], treat_start=30,
                                  inference="bootstrap").ensemble["att_lo"])
    with pytest.raises(ValueError, match="inference"):
        d.evaluate(treated=["M00"], treat_start=48, inference="nope")


def test_evaluate_validates_inputs():
    Y, names = geo_panel(n=8, t=40)
    d = GeoDesign(Y, names=names)
    with pytest.raises(ValueError, match="treat_start"):
        d.evaluate(treated=["M00"], treat_start=0)
    with pytest.raises(ValueError, match="unknown methods"):
        d.evaluate(treated=["M00"], treat_start=30, methods=["SC", "XYZ"])


def test_evaluate_timeline_figure(tmp_path):
    Y, names = geo_panel(n=16, t=60)
    Yt = Y.copy()
    Yt[0, 50:] *= 1.07
    ev = GeoDesign(Yt, names=names).evaluate(treated=["M00"], treat_start=50)
    e = ev.ensemble
    # the timeline arrays are present and sensibly shaped
    assert "full_gap" in e and len(e["full_gap"]) == 60
    assert e["sigma_pre"] >= 0 and e["point_hw"] >= 0
    # pre-period gap is centered (fit residual, not a level offset)
    assert abs(float(np.mean(e["full_gap"][:50]))) < 1e-6
    # cumulative CI present and the band grows with horizon (autocorrelation-aware)
    assert "cum_lo" in e and e["cum_hi"] >= e["cum_lo"]
    w0 = e["cum_hi_curve"][0] - e["cum_curve"][0]
    wN = e["cum_hi_curve"][-1] - e["cum_curve"][-1]
    assert wN >= w0
    out = tmp_path / "tl.png"
    ev.plot_effect_over_time(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_select_include_forces_markets():
    Y, names = geo_panel(n=14)
    d = GeoDesign(Y, names=names)
    ranked = d.select_markets(test_len=10, target_lift=0.1, max_treated=3,
                              n_candidates=30, include=["M05"], top=8)
    assert ranked and all("M05" in c["markets"] for c in ranked)
    assert all(len(c["markets"]) <= 3 for c in ranked)
    # too many forced markets for the size budget
    with pytest.raises(ValueError, match="max_treated"):
        d.select_markets(test_len=10, target_lift=0.1, max_treated=2,
                         include=["M00", "M01", "M02"])


def test_select_exclude_drops_markets():
    Y, names = geo_panel(n=14)
    d = GeoDesign(Y, names=names)
    ranked = d.select_markets(test_len=10, target_lift=0.1, max_treated=3,
                              n_candidates=40, exclude=["M01", "M02"], top=10)
    chosen = {m for c in ranked for m in c["markets"]}
    assert "M01" not in chosen and "M02" not in chosen
    # include + exclude conflict is rejected
    with pytest.raises(ValueError, match="both include and exclude"):
        d.select_markets(test_len=10, target_lift=0.1, max_treated=3,
                         include=["M03"], exclude=["M03"])


def test_power_and_evaluate_exclude():
    Y, names = geo_panel(n=16, t=60)
    d = GeoDesign(Y, names=names)
    rep = d.power(treated=["M00"], test_len=8, lifts=[0.0, 0.1], exclude=["M05", "M06"])
    assert "ENSEMBLE" in rep.results
    with pytest.raises(ValueError, match="excluded"):
        d.power(treated=["M00"], test_len=8, exclude=["M00"])
    Yt = Y.copy()
    Yt[0, 50:] *= 1.08
    ev = GeoDesign(Yt, names=names).evaluate(treated=["M00"], treat_start=50,
                                             exclude=["M07"])
    assert ev.lift == ev.lift  # not NaN


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


def test_from_long_robust_to_messy_dtypes():
    pd = pytest.importorskip("pandas")
    Y, names = geo_panel(n=5, t=10)
    rows = []
    for i, nm in enumerate(names):
        for t in range(Y.shape[1]):
            rows.append({"dma": nm,
                         "week": f"2024-01-{t+1:02d}",     # date strings
                         "sales": repr(float(Y[i, t]))})    # numeric-as-string (full precision)
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # shuffle order
    d = GeoDesign.from_long(df, location="dma", time="week", outcome="sales")
    assert d.n == 5 and d.t == 10
    # columns came back in chronological (date) order, so column 0 == period 0
    np.testing.assert_allclose(d.Y[:, 0], Y[[names.index(x) for x in d.names], 0], rtol=1e-9)


def test_from_long_errors_on_nonnumeric_and_gaps():
    pd = pytest.importorskip("pandas")
    Y, names = geo_panel(n=4, t=6)
    # build the outcome column as object so a string value is allowed in it
    rows = [{"dma": nm, "week": t, "sales": (str(Y[i, t]) if not (i == 0 and t == 3) else "N/A")}
            for i, nm in enumerate(names) for t in range(6)]
    bad = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="non-numeric"):
        GeoDesign.from_long(bad, location="dma", time="week", outcome="sales")
    good_rows = [{"dma": nm, "week": t, "sales": float(Y[i, t])}
                 for i, nm in enumerate(names) for t in range(6)]
    gappy = pd.DataFrame(good_rows).iloc[:-1]         # drop a cell → unbalanced
    with pytest.raises(ValueError, match="unbalanced|missing"):
        GeoDesign.from_long(gappy, location="dma", time="week", outcome="sales")


def test_recommend_sweeps_and_recommends():
    Y, names = geo_panel(n=14)
    d = GeoDesign(Y, names=names)
    grid = d.recommend(test_lengths=[8, 12], n_geos_options=[1, 2, 3],
                       target_lift=0.1, alphas=[0.1], n_candidates=24)
    assert len(grid.rows) >= 1
    rec = grid.recommended
    assert rec is not None
    assert rec["test_len"] in (8, 12) and rec["n_geos"] in (1, 2, 3)
    assert "SPECIFICATION RECOMMENDATIONS" in grid.summary()


def test_recommend_plot_writes_file(tmp_path):
    Y, names = geo_panel(n=12)
    d = GeoDesign(Y, names=names)
    grid = d.recommend(test_lengths=[8, 12], n_geos_options=[1, 2],
                       target_lift=0.1, alphas=[0.05, 0.1], n_candidates=20)
    out = tmp_path / "tradeoffs.png"
    grid.plot(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_alpha_affects_power_threshold():
    Y, names = geo_panel()
    d = GeoDesign(Y, names=names)
    strict = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05], alpha=0.01)
    loose = d.power(treated=["M00"], test_len=10, lifts=[0.0, 0.05], alpha=0.20)
    # A looser alpha makes the critical threshold smaller → power no lower.
    assert loose.best.power[-1] >= strict.best.power[-1] - 1e-9
