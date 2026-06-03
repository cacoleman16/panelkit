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
    def from_long(cls, df, location: str, time: str, outcome: str) -> "GeoDesign":
        """Build from a long DataFrame with one row per (location, time)."""
        wide = df.pivot(index=location, columns=time, values=outcome)
        if wide.isna().any().any():
            raise ValueError(
                "pivoted panel has gaps (some location×time cells missing); "
                "panelkit needs a balanced panel — fill or aggregate first"
            )
        wide = wide.sort_index(axis=1)  # chronological columns
        return cls(wide.to_numpy(), names=list(wide.index))

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
    ) -> _PowerReport:
        """Power analysis for a specified treated-market set across methods."""
        idx = self._resolve(treated)
        names = [self.names[i] for i in idx]
        lifts = list(_DEFAULT_LIFTS if lifts is None else lifts)
        if 0.0 not in lifts:
            lifts = [0.0] + list(lifts)
        lifts = sorted(set(float(x) for x in lifts))
        results = {}
        for m in methods:
            results[m] = _panelkit.geo_power(
                self.Y, idx, int(test_len), lifts, m.lower(), alpha, target_power, 0
            )
        diag = _panelkit.geo_diagnostics(self.Y, idx, int(test_len))
        rec = recommended if recommended in results else list(results)[0]
        return _PowerReport(self, idx, names, test_len, results, diag, rec, alpha, target_power)

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
    ) -> list:
        """Search candidate treatment-market sets and return the top ranked."""
        elig = self._resolve(eligible) if eligible is not None else list(range(self.n))
        ranked = _panelkit.geo_select(
            self.Y, elig, int(max_treated), int(test_len), float(target_lift),
            method.lower(), alpha, target_power, 0, int(n_candidates), int(seed),
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


# --------------------------------------------------------------------------
# Professional plotting.
# --------------------------------------------------------------------------
_PK_BLUE = "#2563eb"
_PK_GREEN = "#059669"
_PK_AMBER = "#d97706"
_PK_GREY = "#9ca3af"
_METHOD_COLORS = {"SC": _PK_GREY, "ASC": _PK_AMBER, "SDID": _PK_BLUE}


def _plot_power(rep: _PowerReport, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
