"""Geo-experiment design: power analysis, market selection, and a plain-English
report with professional graphs — the planning layer in front of a geo test.

    from panelkit.design import GeoDesign

    # from a long/tidy DataFrame (location, date, outcome):
    design = GeoDesign.from_long(df, location="dma", time="week", outcome="sales")
    # or straight from an N×T matrix:
    design = GeoDesign(Y, names=[...])

    rep = design.power(treated=["chicago", "denver"], test_len=4, lifts=[0,.02,.05,.1])
    print(rep.summary())
    rep.plot("power.png")

    ranked = design.select_markets(test_len=4, target_lift=0.05, max_treated=3)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _panelkit

_METHODS = ("SC", "ASC", "SDID")
_DEFAULT_LIFTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


class _PowerReport:
    """Result of a power analysis across methods, with a report and plots."""

    def __init__(self, design, treated_idx, treated_names, test_len, results,
                 diag, recommended, alpha, target_power):
        self._d = design
        self.treated_idx = treated_idx
        self.treated_names = treated_names
        self.test_len = test_len
        self.results = results            # dict: method -> raw PowerResult
        self.diagnostics = diag
        self.recommended = recommended
        self.alpha = alpha
        self.target_power = target_power

    @property
    def best(self):
        return self.results[self.recommended]

    # --- headline numbers (from the recommended method) ---
    @property
    def mde_pct(self):
        return self.best.mde_pct

    @property
    def mde_absolute(self):
        return self.best.mde_abs_per_period

    @property
    def mde_cumulative(self):
        return self.best.mde_cumulative

    @property
    def confidence(self):
        return self.diagnostics.confidence

    def summary(self) -> str:
        d = self.diagnostics
        names = ", ".join(map(str, self.treated_names))
        lines = []
        lines.append("=" * 64)
        lines.append("GEO TEST DESIGN REPORT")
        lines.append("=" * 64)
        lines.append(f"Treatment markets : {names}")
        lines.append(f"Test duration     : {self.test_len} periods")
        lines.append(f"Holdout (exposure): {100*d.holdout_pct:.1f}% of total volume")
        lines.append(f"Design confidence : {d.confidence:.0f}/100")
        lines.append("")
        lines.append(f"Recommended method: {self.recommended}")
        if self.mde_pct is not None:
            lines.append(
                f"Minimum detectable effect (at {int(100*self.target_power)}% power, "
                f"{int(100*(1-self.alpha))}% confidence):"
            )
            lines.append(f"   • {100*self.mde_pct:.2f}% lift")
            lines.append(f"   • {self.mde_absolute:,.1f} per period, per treated market (absolute)")
            lines.append(f"   • {self.mde_cumulative:,.0f} cumulative incremental over the test window")
        else:
            lines.append("Minimum detectable effect: not reached within the tested lift grid "
                         "(design is underpowered — add markets, length, or history).")
        lines.append("")
        lines.append("Method comparison (MDE at target power):")
        for m in self.results:
            r = self.results[m]
            mde = f"{100*r.mde_pct:.2f}%" if r.mde_pct is not None else "—"
            lines.append(f"   {m:<5} MDE {mde:>7}   (pre-fit null SE {r.se_null:.3g}, {r.n_windows} windows)")
        lines.append("")
        lines.append("Diagnostics:")
        lines.append(f"   • Pre-period fit (relative RMSPE): {d.pre_fit_rel:.2f}  "
                     f"({'good' if d.pre_fit_rel < 0.25 else 'fair' if d.pre_fit_rel < 0.5 else 'weak'})")
        lines.append(f"   • Improvement over naive DiD     : {100*d.improvement_vs_naive:.0f}% lower pre-period error")
        lines.append(f"   • Seasonality strength           : {d.seasonality_strength:.2f}")
        lines.append(f"   • Pre-period stability           : {d.stability_score:.2f}")
        if d.warnings:
            lines.append("")
            lines.append("⚠ Warnings:")
            for w in d.warnings:
                lines.append(f"   • {w}")
        lines.append("")
        lines.append(_verdict(d.confidence, self.mde_pct))
        lines.append("=" * 64)
        return "\n".join(lines)

    def plot(self, path: str | None = None):
        """Render the professional design figure. Returns the matplotlib Figure;
        saves to `path` if given."""
        return _plot_power(self, path)

    def __repr__(self):
        mde = f"{100*self.mde_pct:.2f}%" if self.mde_pct is not None else "n/a"
        return (f"PowerReport(treated={self.treated_names}, recommended={self.recommended}, "
                f"MDE={mde}, confidence={self.confidence:.0f})")


def _verdict(confidence, mde_pct):
    if mde_pct is None:
        return "VERDICT: ✗ Underpowered — this design can't reliably detect a realistic lift."
    if confidence >= 75 and mde_pct <= 0.05:
        return ("VERDICT: ✓ Strong design — well-powered for small lifts with a "
                "trustworthy counterfactual.")
    if confidence >= 60:
        return ("VERDICT: ~ Workable design — usable, but watch the warnings and "
                "consider a larger lift target or more markets.")
    return ("VERDICT: ✗ Risky design — low confidence; revisit market choice, "
            "history length, or holdout size before spending.")


class _DiagnosticsReport:
    """Real-world guardrails for a design, with a summary and a visual."""

    def __init__(self, treated_names, t0, test_len, diag, treated_series, synthetic):
        self.treated_names = treated_names
        self.t0 = t0
        self.test_len = test_len
        self._raw = diag
        self.treated_series = np.asarray(treated_series, dtype=float)
        self.synthetic = np.asarray(synthetic, dtype=float)

    @property
    def holdout_pct(self):
        return self._raw.holdout_pct

    @property
    def confidence(self):
        return self._raw.confidence

    @property
    def warnings(self):
        return list(self._raw.warnings)

    def summary(self) -> str:
        d = self._raw
        lines = ["GUARDRAILS — " + ", ".join(map(str, self.treated_names))]
        lines.append(f"  holdout            : {100*d.holdout_pct:.1f}% of volume")
        lines.append(f"  pre-period fit     : rel. RMSPE {d.pre_fit_rel:.2f} "
                     f"({'good' if d.pre_fit_rel < 0.25 else 'fair' if d.pre_fit_rel < 0.5 else 'weak'})")
        lines.append(f"  improvement v naive: {100*d.improvement_vs_naive:.0f}%")
        lines.append(f"  seasonality        : {d.seasonality_strength:.2f}")
        lines.append(f"  stability          : {d.stability_score:.2f}")
        lines.append(f"  confidence         : {d.confidence:.0f}/100")
        if d.warnings:
            lines.append("  warnings:")
            for w in d.warnings:
                lines.append(f"    ⚠ {w}")
        else:
            lines.append("  ✓ no warnings")
        return "\n".join(lines)

    def plot(self, path: str | None = None):
        """Render the guardrails figure. Returns the matplotlib Figure."""
        return _plot_guardrails(self, path)

    def __repr__(self):
        return (f"GuardrailsReport(confidence={self.confidence:.0f}, "
                f"holdout={100*self.holdout_pct:.1f}%, warnings={len(self.warnings)})")


class GeoDesign:
    """A geo panel ready for power analysis and market selection.

    Construct from an ``N×T`` matrix (``GeoDesign(Y, names=...)``) or from a
    long/tidy DataFrame (``GeoDesign.from_long(df, location, time, outcome)``).
    """

    def __init__(self, Y, names: Sequence | None = None):
        arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
        if arr.ndim != 2:
            raise ValueError(f"Y must be 2-D (N markets × T periods), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Y has NaN/inf; the panel must be complete and finite")
        self.Y = arr
        self.n, self.t = arr.shape
        self.names = list(names) if names is not None else list(range(self.n))
        if len(self.names) != self.n:
            raise ValueError(f"names length {len(self.names)} != n markets {self.n}")
        self._index = {nm: i for i, nm in enumerate(self.names)}

    @classmethod
    def from_long(cls, df, location: str, time: str, outcome: str, *, agg: str = "sum"):
        """Build a :class:`GeoDesign` from a long/tidy DataFrame with one row per
        (location, time).

        Robust to messy real-world dtypes — you do **not** need to pre-clean:

        - **outcome** is coerced to numeric; truly non-numeric values (e.g.
          ``"1,234"`` with thousands separators that won't parse, or ``"N/A"``)
          raise a clear error pointing at the offending value.
        - **time** is parsed as datetime when possible (so ``"2024-01-07"`` and
          ``"2024-01-14"`` order correctly), else as numbers, else sorted as
          strings — columns always end up in true chronological order.
        - **location** is cast to string for stable market names.
        - **duplicate** (location, time) rows are aggregated (``agg``, default
          ``"sum"``) with a warning rather than silently dropped.
        - a **gappy** panel (some location×time cells missing) raises an error
          telling you how many cells are missing.

        Parameters
        ----------
        df : pandas.DataFrame
        location, time, outcome : str
            Column names for the market id, the period, and the metric.
        agg : {"sum", "mean", "first"}
            How to combine duplicate (location, time) rows.
        """
        import warnings

        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("from_long requires pandas (`pip install pandas`)") from e

        for col in (location, time, outcome):
            if col not in df.columns:
                raise ValueError(f"column {col!r} not found; DataFrame has {list(df.columns)}")

        d = df[[location, time, outcome]].copy()

        # --- outcome → numeric -------------------------------------------------
        num = pd.to_numeric(d[outcome], errors="coerce")
        newly_nan = num.isna() & ~d[outcome].isna()
        if newly_nan.any():
            bad = d.loc[newly_nan, outcome].iloc[0]
            raise ValueError(
                f"{int(newly_nan.sum())} non-numeric value(s) in outcome column "
                f"{outcome!r} (e.g. {bad!r}); clean these before fitting"
            )
        if num.isna().any():
            raise ValueError(
                f"{int(num.isna().sum())} missing outcome value(s) in {outcome!r}; "
                "panelkit needs a complete panel"
            )
        d[outcome] = num.astype(float)

        # --- location → string -------------------------------------------------
        d[location] = d[location].astype(str)

        # --- time → a sortable ordering ---------------------------------------
        raw_time = d[time]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # we handle the fallback ourselves
            parsed = pd.to_datetime(raw_time, errors="coerce")
        if parsed.notna().all():
            order_key = parsed
        else:
            as_num = pd.to_numeric(raw_time, errors="coerce")
            order_key = as_num if as_num.notna().all() else raw_time.astype(str)
        d["__order__"] = order_key
        # stable label for the column (keep original time value)
        d["__time__"] = raw_time

        # --- dedupe duplicate (location, time) --------------------------------
        dup = d.duplicated([location, "__time__"]).sum()
        if dup:
            warnings.warn(
                f"{int(dup)} duplicate (location, time) rows aggregated with agg={agg!r}; "
                "pre-aggregate if that's not what you want.",
                stacklevel=2,
            )
            grp = d.groupby([location, "__time__"], as_index=False)
            d = grp.agg({outcome: agg, "__order__": "first"})

        # --- order the time axis, pivot ---------------------------------------
        time_order = (
            d[["__time__", "__order__"]]
            .drop_duplicates()
            .sort_values("__order__")["__time__"]
            .tolist()
        )
        wide = d.pivot(index=location, columns="__time__", values=outcome)
        wide = wide.reindex(columns=time_order)

        n_missing = int(wide.isna().to_numpy().sum())
        if n_missing:
            total = wide.shape[0] * wide.shape[1]
            raise ValueError(
                f"unbalanced panel: {n_missing} of {total} (market × period) cells are "
                "missing after pivoting. panelkit needs a balanced panel — fill, drop, "
                "or aggregate the gaps first."
            )
        return cls(wide.to_numpy(dtype=float), names=[str(i) for i in wide.index])

    def _resolve(self, markets) -> list[int]:
        out = []
        for m in markets:
            if isinstance(m, (int, np.integer)) and not isinstance(m, bool):
                idx = int(m)
                if not (0 <= idx < self.n):
                    raise ValueError(f"market index {idx} out of range [0, {self.n})")
                out.append(idx)
            else:
                if m not in self._index:
                    raise ValueError(f"unknown market name {m!r}")
                out.append(self._index[m])
        return out

    def power(
        self,
        treated,
        test_len: int,
        lifts: Sequence[float] | None = None,
        methods: Sequence[str] = _METHODS,
        alpha: float = 0.10,
        target_power: float = 0.80,
        recommended: str = "SDID",
        lookback: int | None = None,
    ) -> _PowerReport:
        """Power analysis for a specified treated-market set across methods.

        Powers over many historical placebo windows (sliding the test window
        across history); ``lookback=k`` restricts to the most-recent ``k`` windows,
        which are most representative of the upcoming test."""
        idx = self._resolve(treated)
        names = [self.names[i] for i in idx]
        lifts = list(_DEFAULT_LIFTS if lifts is None else lifts)
        if 0.0 not in lifts:
            lifts = [0.0] + list(lifts)
        lifts = sorted(set(float(x) for x in lifts))
        lb = None if lookback is None else int(lookback)
        results = {}
        for m in methods:
            results[m] = _panelkit.geo_power(
                self.Y, idx, int(test_len), lifts, m.lower(), alpha, target_power, 0, lb
            )
        diag = _panelkit.geo_diagnostics(self.Y, idx, int(test_len))
        rec = recommended if recommended in results else list(results)[0]
        return _PowerReport(self, idx, names, test_len, results, diag, rec, alpha, target_power)

    def diagnose(self, treated, test_len: int) -> "_DiagnosticsReport":
        """Real-world guardrails for a treated-market set: pre-period fit,
        seasonality, holdout, stability, and warnings — with a visual.

        Returns a report with ``.summary()`` and ``.plot(path)`` (the guardrails
        figure: treated-vs-synthetic pre-fit, seasonality ACF, holdout share, and
        a scorecard listing any warnings)."""
        idx = self._resolve(treated)
        names = [self.names[i] for i in idx]
        t0 = self.t - int(test_len)
        diag = _panelkit.geo_diagnostics(self.Y, idx, int(test_len))
        # Treated-average series and the SC counterfactual (from the SC weights).
        treated_series = self.Y[idx].mean(axis=0)
        scres = _panelkit.fit_sc(self.Y, idx, int(t0), 0.0, False, 0.95)
        w = np.asarray(scres.weights, dtype=float)
        donors = np.asarray(scres.donor_ids, dtype=int)
        synthetic = self.Y[donors].T @ w if len(donors) else np.full(self.t, np.nan)
        return _DiagnosticsReport(names, t0, test_len, diag, treated_series, synthetic)

    def select_markets(
        self,
        test_len: int,
        target_lift: float,
        max_treated: int = 3,
        eligible=None,
        method: str = "SDID",
        alpha: float = 0.10,
        target_power: float = 0.80,
        n_candidates: int = 200,
        seed: int = 0,
        top: int = 10,
        exact_size: int | None = None,
        lookback: int | None = None,
    ) -> list:
        """Search candidate treatment-market sets and return the top ranked.

        ``exact_size=k`` restricts the search to sets of exactly ``k`` markets
        (otherwise sizes 1..``max_treated`` are considered). ``lookback=k`` powers
        over the most-recent ``k`` historical windows."""
        elig = self._resolve(eligible) if eligible is not None else list(range(self.n))
        ranked = _panelkit.geo_select(
            self.Y, elig, int(max_treated), int(test_len), float(target_lift),
            method.lower(), alpha, target_power, 0, int(n_candidates), int(seed),
            None if exact_size is None else int(exact_size),
            None if lookback is None else int(lookback),
        )
        out = []
        for c in ranked[:top]:
            out.append({
                "markets": [self.names[i] for i in c.treated],
                "power_at_target": c.power_at_target,
                "mde_pct": c.mde_pct,
                "holdout_pct": c.holdout_pct,
                "pre_fit_rel": c.pre_fit_rel,
                "confidence": c.confidence,
                "score": c.score,
            })
        return out

    def recommend(
        self,
        test_lengths: Sequence[int],
        n_geos_options: Sequence[int],
        target_lift: float,
        alphas: Sequence[float] = (0.10,),
        eligible=None,
        method: str = "SDID",
        target_power: float = 0.80,
        n_candidates: int = 80,
        seed: int = 0,
        min_confidence: float = 60.0,
        lookback: int | None = None,
    ) -> "_ScenarioGrid":
        """Sweep designs across **specifications** — test length × number of geos
        × significance level (alpha) — and recommend the best.

        For each (alpha, test_len, n_geos) cell it searches for the best set of
        exactly ``n_geos`` treatment markets and records its MDE, power, holdout,
        and confidence. Returns a :class:`_ScenarioGrid` with a recommendation,
        a plain-English summary, and a tradeoffs figure.
        """
        rows = []
        for alpha in alphas:
            for tl in test_lengths:
                for ng in n_geos_options:
                    ranked = self.select_markets(
                        test_len=tl, target_lift=target_lift, max_treated=ng,
                        eligible=eligible, method=method, alpha=alpha,
                        target_power=target_power, n_candidates=n_candidates,
                        seed=seed, top=1, exact_size=ng, lookback=lookback,
                    )
                    best = ranked[0] if ranked else None
                    if best is None:
                        continue
                    rows.append({
                        "alpha": float(alpha),
                        "test_len": int(tl),
                        "n_geos": int(ng),
                        "markets": best["markets"],
                        "mde_pct": best["mde_pct"],
                        "power_at_target": best["power_at_target"],
                        "holdout_pct": best["holdout_pct"],
                        "confidence": best["confidence"],
                    })
        return _ScenarioGrid(rows, target_lift, target_power, list(alphas),
                             list(test_lengths), list(n_geos_options), min_confidence)


class _ScenarioGrid:
    """Recommendations swept across test length, number of geos, and alpha."""

    def __init__(self, rows, target_lift, target_power, alphas, test_lengths,
                 n_geos_options, min_confidence):
        self.rows = rows
        self.target_lift = target_lift
        self.target_power = target_power
        self.alphas = sorted(alphas)
        self.test_lengths = sorted(test_lengths)
        self.n_geos_options = sorted(n_geos_options)
        self.min_confidence = min_confidence

    @property
    def recommended(self):
        """The recommended specification: smallest MDE among trustworthy designs
        (confidence ≥ min_confidence), breaking ties toward shorter tests and
        fewer geos. Falls back to the lowest-MDE design if none clear the bar."""
        usable = [r for r in self.rows if r["mde_pct"] is not None]
        if not usable:
            return None
        trustworthy = [r for r in usable if r["confidence"] >= self.min_confidence]
        pool = trustworthy or usable
        return min(pool, key=lambda r: (r["mde_pct"], r["test_len"], r["n_geos"]))

    def table(self):
        return list(self.rows)

    def summary(self) -> str:
        rec = self.recommended
        lines = ["=" * 64, "SPECIFICATION RECOMMENDATIONS", "=" * 64]
        lines.append(f"Swept: test_len {self.test_lengths} × n_geos {self.n_geos_options} "
                     f"× alpha {self.alphas}  ({len(self.rows)} designs)")
        lines.append(f"Detecting a {100*self.target_lift:.0f}% lift at "
                     f"{int(100*self.target_power)}% power.")
        lines.append("")
        if rec is None:
            lines.append("No specification reached the target power within the grid — "
                         "extend history, add geos, or accept a larger target lift.")
            lines.append("=" * 64)
            return "\n".join(lines)
        mde = f"{100*rec['mde_pct']:.2f}%" if rec["mde_pct"] is not None else "—"
        lines.append("RECOMMENDED DESIGN:")
        lines.append(f"   • {rec['n_geos']} geos, {rec['test_len']}-period test, "
                     f"alpha {rec['alpha']:.2f}")
        lines.append(f"   • Markets: {', '.join(map(str, rec['markets']))}")
        lines.append(f"   • MDE {mde}  ·  confidence {rec['confidence']:.0f}/100  ·  "
                     f"holdout {100*rec['holdout_pct']:.1f}%")
        lines.append("")
        lines.append("Top alternatives (by MDE, trustworthy designs):")
        usable = [r for r in self.rows
                  if r["mde_pct"] is not None and r["confidence"] >= self.min_confidence]
        usable.sort(key=lambda r: r["mde_pct"])
        for r in usable[:5]:
            lines.append(f"   {r['n_geos']}g × {r['test_len']}p @α{r['alpha']:.2f}: "
                         f"MDE {100*r['mde_pct']:.2f}%  conf {r['confidence']:.0f}  "
                         f"holdout {100*r['holdout_pct']:.1f}%")
        lines.append("")
        lines.append("Read the tradeoff figure: longer tests and more geos lower the "
                     "detectable lift, but cost more holdout/time — pick the knee.")
        lines.append("=" * 64)
        return "\n".join(lines)

    def plot(self, path: str | None = None):
        """Render the specification-tradeoffs figure. Returns the Figure."""
        return _plot_scenarios(self, path)

    def __repr__(self):
        rec = self.recommended
        if rec is None:
            return "ScenarioGrid(no powered design)"
        return (f"ScenarioGrid(recommended={rec['n_geos']}g×{rec['test_len']}p"
                f"@α{rec['alpha']:.2f}, MDE={100*rec['mde_pct']:.2f}%)")


# --------------------------------------------------------------------------
# Professional plotting.
# --------------------------------------------------------------------------
_PK_BLUE = "#2563eb"
_PK_GREEN = "#059669"
_PK_AMBER = "#d97706"
_PK_GREY = "#9ca3af"
_METHOD_COLORS = {"SC": _PK_GREY, "ASC": _PK_AMBER, "SDID": _PK_BLUE}


def _require_mpl():
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib — install it with `pip install panelkit[plot]`"
        ) from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _plot_power(rep: _PowerReport, path):
    _, plt = _require_mpl()
    from matplotlib.gridspec import GridSpec

    best = rep.best
    fig = plt.figure(figsize=(11, 7.2))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.26)

    # Panel 1: power curves (all methods).
    ax = fig.add_subplot(gs[0, :])
    for m, r in rep.results.items():
        x = [100 * l for l in r.lifts]
        ax.plot(x, r.power, "o-", color=_METHOD_COLORS.get(m, _PK_GREY),
                lw=2.4 if m == rep.recommended else 1.6,
                alpha=1.0 if m == rep.recommended else 0.75,
                markersize=5, label=f"{m}{'  ★' if m == rep.recommended else ''}")
    ax.axhline(rep.target_power, ls="--", color="#374151", lw=1.0,
               label=f"target power {int(100*rep.target_power)}%")
    if best.mde_pct is not None:
        mx = 100 * best.mde_pct
        ax.axvline(mx, ls=":", color=_PK_GREEN, lw=1.6)
        ax.annotate(f"MDE ≈ {mx:.1f}%", (mx, rep.target_power),
                    textcoords="offset points", xytext=(8, -18),
                    color=_PK_GREEN, fontweight="bold")
    ax.set_xlabel("true lift (%)")
    ax.set_ylabel("power (detection rate)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Power curve — treated: {', '.join(map(str, rep.treated_names))}  "
                 f"({rep.test_len}-period test)", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)

    # Panel 2: estimated vs true lift, with CI band (recommended method).
    ax2 = fig.add_subplot(gs[1, 0])
    x = [100 * l for l in best.lifts]
    est = [100 * e for e in best.est_mean]
    lo = [100 * e for e in best.est_lo]
    hi = [100 * e for e in best.est_hi]
    ax2.plot(x, x, ls="--", color=_PK_GREY, lw=1.0, label="truth")
    ax2.fill_between(x, lo, hi, color=_PK_BLUE, alpha=0.18, label=f"{int(100*(1-rep.alpha))}% CI")
    ax2.plot(x, est, "o-", color=_PK_BLUE, lw=2.0, markersize=4, label=f"{rep.recommended} estimate")
    ax2.set_xlabel("true lift (%)")
    ax2.set_ylabel("estimated lift (%)")
    ax2.set_title("Estimate accuracy & uncertainty", fontweight="bold")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel 3: confidence gauge + diagnostics bars.
    ax3 = fig.add_subplot(gs[1, 1])
    d = rep.diagnostics
    metrics = [
        ("confidence", d.confidence / 100.0),
        ("pre-fit", max(0.0, 1.0 - d.pre_fit_rel)),
        ("vs naive", d.improvement_vs_naive),
        ("stability", d.stability_score),
    ]
    labels = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = [_PK_GREEN if v >= 0.66 else _PK_AMBER if v >= 0.4 else "#dc2626" for v in vals]
    ax3.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    ax3.set_xlim(0, 1)
    ax3.set_title(f"Design quality — confidence {d.confidence:.0f}/100", fontweight="bold")
    ax3.grid(True, axis="x", alpha=0.25)
    for i, v in enumerate(vals[::-1]):
        ax3.text(min(v + 0.02, 0.92), i, f"{v:.2f}", va="center", fontsize=8)

    fig.suptitle("panelkit · geo test design", fontsize=13, fontweight="bold", x=0.01, ha="left")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# Distinct, colorblind-friendly line colors (one per #geos), not a gradient.
_GEO_PALETTE = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]


def _plot_scenarios(grid: "_ScenarioGrid", path):
    _, plt = _require_mpl()
    import numpy as _np
    from matplotlib.gridspec import GridSpec

    a0 = grid.alphas[0]                       # primary alpha for the main panels
    rec = grid.recommended
    by = {(r["alpha"], r["test_len"], r["n_geos"]): r for r in grid.rows}
    color_for = {ng: _GEO_PALETTE[i % len(_GEO_PALETTE)]
                 for i, ng in enumerate(grid.n_geos_options)}

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(12, 7.6))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.30)

    # ---- Panel 1: MDE vs test length, one labelled line per #geos. ----
    ax = fig.add_subplot(gs[0, :])
    ymax = 0.0
    for ng in grid.n_geos_options:
        xs, ys = [], []
        for tl in grid.test_lengths:
            r = by.get((a0, tl, ng))
            if r and r["mde_pct"] is not None:
                xs.append(tl)
                ys.append(100 * r["mde_pct"])
        if not xs:
            continue
        ymax = max(ymax, max(ys))
        c = color_for[ng]
        ax.plot(xs, ys, "o-", color=c, lw=2.6, markersize=7, label=f"{ng} geos", zorder=3)
        # label each line at its right end so you don't need to trace the legend
        ax.annotate(f"{ng} geos", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", color=c, fontweight="bold", fontsize=10)
    tgt = 100 * grid.target_lift
    ax.axhline(tgt, ls="--", color="#374151", lw=1.2)
    ax.axhspan(tgt, max(ymax, tgt) * 1.08 + 0.5, color="#fca5a5", alpha=0.12)
    ax.annotate("can't detect below this lift", (grid.test_lengths[0], tgt),
                textcoords="offset points", xytext=(4, 6), color="#b91c1c", fontsize=9)
    if rec is not None and rec["alpha"] == a0 and rec["mde_pct"] is not None:
        ax.plot(rec["test_len"], 100 * rec["mde_pct"], "*", color="#111827",
                markersize=22, zorder=6)
        ax.annotate("recommended", (rec["test_len"], 100 * rec["mde_pct"]),
                    textcoords="offset points", xytext=(6, -16), fontweight="bold")
    ax.set_xlabel("test length (periods)")
    ax.set_ylabel("min. detectable lift (%)  ·  lower = better")
    ax.set_title(f"How small a lift can you detect?  (α = {a0:.2f})", fontweight="bold")
    ax.set_xticks(grid.test_lengths)
    ax.set_ylim(0, max(ymax, tgt) * 1.12 + 0.5)
    ax.margins(x=0.08)
    ax.grid(True, alpha=0.25)
    # endpoint labels already identify lines; keep the legend out of the way
    # (lower-left, where the curves don't go).
    ax.legend(title="treatment markets", loc="lower left", framealpha=0.95, ncol=2,
              fontsize=9)
    ax.margins(x=0.12)  # room for the right-edge endpoint labels

    # ---- Panel 2: MDE heatmap (red = worse, green = better), readable text. ----
    ax2 = fig.add_subplot(gs[1, 0])
    grid_mde = _np.full((len(grid.n_geos_options), len(grid.test_lengths)), _np.nan)
    for i, ng in enumerate(grid.n_geos_options):
        for k, tl in enumerate(grid.test_lengths):
            r = by.get((a0, tl, ng))
            if r and r["mde_pct"] is not None:
                grid_mde[i, k] = 100 * r["mde_pct"]
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#e5e7eb")  # grey for un-powered cells
    finite = grid_mde[_np.isfinite(grid_mde)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    im = ax2.imshow(grid_mde, aspect="auto", cmap=cmap, origin="lower",
                    vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(len(grid.test_lengths)))
    ax2.set_xticklabels(grid.test_lengths)
    ax2.set_yticks(range(len(grid.n_geos_options)))
    ax2.set_yticklabels(grid.n_geos_options)
    ax2.set_xlabel("test length")
    ax2.set_ylabel("number of geos")
    ax2.set_title("Detectable lift (%) by design", fontweight="bold")
    span = (vmax - vmin) or 1.0
    for i in range(grid_mde.shape[0]):
        for k in range(grid_mde.shape[1]):
            v = grid_mde[i, k]
            if not _np.isnan(v):
                r_, g_, b_, _ = cmap((v - vmin) / span)
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                ax2.text(k, i, f"{v:.1f}", ha="center", va="center",
                         color="black" if lum > 0.55 else "white",
                         fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label(
        "MDE (%) — greener is better", fontsize=9)

    # ---- Panel 3: alpha sensitivity (recommended spec), else confidence. ----
    ax3 = fig.add_subplot(gs[1, 1])
    if len(grid.alphas) > 1 and rec is not None:
        xs, ys = [], []
        for a in grid.alphas:
            r = by.get((a, rec["test_len"], rec["n_geos"]))
            if r and r["mde_pct"] is not None:
                xs.append(a)
                ys.append(100 * r["mde_pct"])
        ax3.plot(xs, ys, "o-", color=_PK_BLUE, lw=2.6, markersize=7)
        for xa, ya in zip(xs, ys):
            ax3.annotate(f"{ya:.1f}%", (xa, ya), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
        ax3.set_xlabel("significance level α")
        ax3.set_ylabel("min. detectable lift (%)")
        ax3.set_title(f"Looser α → smaller MDE  ({rec['n_geos']}g × {rec['test_len']}p)",
                      fontweight="bold")
        ax3.margins(x=0.15, y=0.2)
    else:
        for ng in grid.n_geos_options:
            xs, ys = [], []
            for tl in grid.test_lengths:
                r = by.get((a0, tl, ng))
                if r:
                    xs.append(tl)
                    ys.append(r["confidence"])
            if xs:
                ax3.plot(xs, ys, "o-", lw=2.2, markersize=6,
                         color=color_for[ng], label=f"{ng} geos")
        ax3.axhline(grid.min_confidence, ls=":", color="#dc2626", lw=1.2,
                    label="min confidence")
        ax3.set_xlabel("test length")
        ax3.set_ylabel("design confidence (0–100)")
        ax3.legend(fontsize=8, framealpha=0.95)
        ax3.set_title("Design confidence by spec", fontweight="bold")
    ax3.grid(True, alpha=0.25)

    fig.suptitle("panelkit · specification tradeoffs", fontsize=14, fontweight="bold",
                 x=0.012, ha="left")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def _plot_guardrails(rep: "_DiagnosticsReport", path):
    _, plt = _require_mpl()
    import numpy as _np
    from matplotlib.gridspec import GridSpec

    d = rep._raw
    t0 = rep.t0
    T = len(rep.treated_series)
    x = _np.arange(T)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(12, 7.8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.26)

    # ---- A: pre-period fit — treated vs synthetic control. ----
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(t0 - 0.5, T - 0.5, color="#dbeafe", alpha=0.5, label="test window")
    ax.plot(x, rep.treated_series, color="#111827", lw=2.2, label="treated (actual)")
    if _np.isfinite(rep.synthetic).all():
        ax.plot(x, rep.synthetic, color="#2563eb", lw=2.0, ls="--",
                label="synthetic control")
    ax.axvline(t0 - 0.5, color="#374151", lw=1.0, ls=":")
    fit_word = "good" if d.pre_fit_rel < 0.25 else "fair" if d.pre_fit_rel < 0.5 else "weak"
    fit_color = "#059669" if d.pre_fit_rel < 0.25 else "#d97706" if d.pre_fit_rel < 0.5 else "#dc2626"
    ax.set_title("Pre-period fit: does the synthetic control track the treated markets?",
                 fontweight="bold")
    ax.set_xlabel("period")
    ax.set_ylabel("outcome")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.annotate(f"pre-fit: {fit_word}  (rel. RMSPE {d.pre_fit_rel:.2f})",
                xy=(0.99, 0.04), xycoords="axes fraction", ha="right",
                color=fit_color, fontweight="bold", fontsize=10)

    # ---- B: seasonality — ACF of pre-period first differences. ----
    axb = fig.add_subplot(gs[1, 0])
    pre = rep.treated_series[:t0]
    dd = _np.diff(pre)
    dd = dd - dd.mean()
    denom = (dd ** 2).sum()
    max_lag = int(min(len(dd) // 2, 26))
    lags = list(range(1, max(max_lag, 2)))
    acf = [float((dd[lag:] * dd[:-lag]).sum() / denom) if denom > 0 else 0.0 for lag in lags]
    best_lag = lags[int(_np.argmax(acf))] if acf else 0
    colors = ["#dc2626" if (lg == best_lag and d.seasonality_strength > 0.3) else "#93c5fd"
              for lg in lags]
    axb.bar(lags, acf, color=colors)
    axb.axhline(0, color="#374151", lw=0.8)
    axb.set_xlabel("lag (periods)")
    axb.set_ylabel("autocorrelation")
    seas_word = ("strong" if d.seasonality_strength > 0.5 else
                 "some" if d.seasonality_strength > 0.3 else "weak")
    title = f"Seasonality: {seas_word} (strength {d.seasonality_strength:.2f})"
    if d.seasonality_strength > 0.3 and best_lag:
        title += f", ≈{best_lag}-period cycle"
    axb.set_title(title, fontweight="bold")
    axb.grid(True, axis="y", alpha=0.25)

    # ---- C: holdout share. ----
    axc = fig.add_subplot(gs[1, 1])
    h = d.holdout_pct
    in_band = 0.03 <= h <= 0.35
    bar_color = "#059669" if in_band else "#d97706"
    axc.barh([0], [100 * h], color=bar_color, height=0.5, label="treated")
    axc.barh([0], [100 * (1 - h)], left=[100 * h], color="#e5e7eb", height=0.5,
             label="control / donors")
    axc.axvspan(3, 35, color="#bbf7d0", alpha=0.35)  # healthy band
    axc.set_xlim(0, 100)
    axc.set_yticks([])
    axc.set_xlabel("% of total volume")
    axc.set_title(f"Holdout: treated = {100*h:.1f}% of volume "
                  f"({'healthy' if in_band else 'check'})", fontweight="bold")
    axc.annotate(f"{100*h:.1f}%", (100 * h / 2, 0), ha="center", va="center",
                 color="white", fontweight="bold")
    axc.legend(loc="lower right", fontsize=8, framealpha=0.95)
    axc.annotate("healthy 3–35%", (19, 0.32), ha="center", color="#15803d", fontsize=8)

    # ---- Warnings / verdict banner across the bottom. ----
    warns = list(d.warnings)
    if warns:
        txt = "⚠ Guardrail warnings:\n" + "\n".join(f"  • {w}" for w in warns)
        box = dict(boxstyle="round,pad=0.5", fc="#fef3c7", ec="#d97706")
    else:
        txt = "✓ No guardrail warnings — design looks clean."
        box = dict(boxstyle="round,pad=0.5", fc="#dcfce7", ec="#059669")
    fig.text(0.012, -0.02, txt, ha="left", va="top", fontsize=9, bbox=box, wrap=True)

    fig.suptitle(f"panelkit · guardrails — confidence {d.confidence:.0f}/100",
                 fontsize=14, fontweight="bold", x=0.012, ha="left")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
