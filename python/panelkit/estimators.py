"""sklearn-style estimator classes wrapping the compiled Rust core.

Each estimator takes an ``N×T`` outcome array (rows = units, columns = time
periods), a list of treated unit indices, and the first post-treatment period.
``.fit(...)`` returns a result object exposing ``.att``, ``.att_path``,
``.weights``, ``.counterfactual``, and (when inference is requested) ``.p_value``
and ``.inference_distribution``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _panelkit


class _Result:
    """Thin Python wrapper over a Rust result object, adding numpy arrays and a
    readable summary."""

    def __init__(self, raw):
        self._raw = raw

    @property
    def att(self) -> float:
        return self._raw.att

    @property
    def att_path(self) -> np.ndarray:
        return np.asarray(self._raw.att_path, dtype=float)

    @property
    def counterfactual(self) -> np.ndarray:
        return np.asarray(self._raw.counterfactual, dtype=float)

    @property
    def treated_post(self) -> np.ndarray:
        return np.asarray(self._raw.treated_post, dtype=float)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self._raw.weights, dtype=float)

    @property
    def donor_ids(self) -> np.ndarray:
        return np.asarray(self._raw.donor_ids, dtype=int)

    @property
    def pre_rmspe(self) -> float:
        return self._raw.pre_rmspe

    @property
    def post_rmspe(self) -> float:
        return self._raw.post_rmspe

    @property
    def p_value(self):
        return self._raw.p_value

    @property
    def se(self):
        return self._raw.se

    @property
    def inference_distribution(self):
        dist = self._raw.inference_distribution
        return None if dist is None else np.asarray(dist, dtype=float)

    def summary(self) -> str:
        lines = [
            f"ATT            : {self.att:.6g}",
            f"pre-RMSPE      : {self.pre_rmspe:.6g}",
            f"post-RMSPE     : {self.post_rmspe:.6g}",
            f"# donors       : {len(self.donor_ids)}",
        ]
        if self.p_value is not None:
            lines.append(f"placebo p-value: {self.p_value:.4g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return repr(self._raw)


def _as_matrix(y) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"y must be 2-D (N units × T periods), got shape {arr.shape}")
    return arr


class SyntheticControl:
    """Synthetic Control (Abadie, Diamond & Hainmueller 2010).

    Parameters
    ----------
    ridge:
        Ridge penalty on the donor weights (0.0 = classic SC).
    inference:
        ``"placebo"`` to run the in-space placebo test, or ``None``.
    level:
        Confidence level for reported intervals.
    """

    def __init__(self, ridge: float = 0.0, inference: str | None = None, level: float = 0.95):
        self.ridge = ridge
        self.inference = inference
        self.level = level

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        treated = [int(t) for t in treated]
        do_placebo = self.inference == "placebo"
        raw = _panelkit.fit_sc(
            mat,
            treated,
            int(treat_time),
            self.ridge,
            do_placebo,
            self.level,
        )
        return _Result(raw)


class AugmentedSC:
    """Augmented Synthetic Control (Ben-Michael, Feller & Rothstein 2021).

    Corrects residual pre-treatment imbalance with a ridge outcome model.
    """

    def __init__(self, sc_ridge: float = 0.0, aug_lambda: float | None = None):
        self.sc_ridge = sc_ridge
        self.aug_lambda = aug_lambda

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        raw = _panelkit.fit_asc(
            mat, [int(t) for t in treated], int(treat_time), self.sc_ridge, self.aug_lambda
        )
        return _Result(raw)


class SyntheticDiD:
    """Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).

    The recommended general-purpose default: unit + time weights feeding a
    doubly-weighted 2×2 difference-in-differences.
    """

    def __init__(self, zeta_scale: float = 1.0):
        self.zeta_scale = zeta_scale

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        raw = _panelkit.fit_sdid(
            mat, [int(t) for t in treated], int(treat_time), self.zeta_scale
        )
        return _Result(raw)


class MCNNM:
    """Matrix-Completion NNM (Athey et al. 2021).

    Estimates a low-rank counterfactual by iterative singular-value
    thresholding (SoftImpute). ``lambda_`` is chosen by cross-validation when
    left as ``None``.
    """

    def __init__(
        self,
        lambda_: float | None = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        seed: int = 0,
    ):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        raw = _panelkit.fit_mcnnm(
            mat,
            [int(t) for t in treated],
            int(treat_time),
            self.lambda_,
            int(self.max_iter),
            float(self.tol),
            int(self.seed),
        )
        return _Result(raw)


class _DiDResult:
    """Result of a difference-in-differences fit, with an event-study path."""

    def __init__(self, raw):
        self._raw = raw

    @property
    def att(self) -> float:
        return self._raw.att

    @property
    def se(self) -> float:
        return self._raw.se

    @property
    def event_time(self) -> np.ndarray:
        return np.asarray(self._raw.event_time, dtype=int)

    @property
    def event_att(self) -> np.ndarray:
        return np.asarray(self._raw.event_att, dtype=float)

    @property
    def event_se(self) -> np.ndarray:
        return np.asarray(self._raw.event_se, dtype=float)

    def summary(self) -> str:
        lines = [f"overall ATT : {self.att:.6g}  (se {self.se:.4g})"]
        if len(self.event_time):
            lines.append("event study:")
            for e, a, s in zip(self.event_time, self.event_att, self.event_se):
                lines.append(f"  e={e:>3d}: {a:>9.4f}  (se {s:.4f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return repr(self._raw)


def _cohorts(treat_start, n) -> list:
    """Normalize a per-unit treatment-start spec to int cohorts (<0 = never)."""
    out = []
    for c in treat_start:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            out.append(-1)
        else:
            out.append(int(c))
    if len(out) != n:
        raise ValueError(f"treat_start length {len(out)} != n_units {n}")
    return out


class TWFE:
    """Two-way fixed-effects DiD with cluster-robust (by unit) SE.

    Note: biased under staggered adoption with heterogeneous effects; prefer
    :class:`CallawaySantAnna` or :class:`SunAbraham` there.
    """

    def fit(self, y, treat_start: Sequence) -> _DiDResult:
        mat = _as_matrix(y)
        return _DiDResult(_panelkit.fit_twfe_py(mat, _cohorts(treat_start, mat.shape[0])))


class CallawaySantAnna:
    """Callaway & Sant'Anna (2021) group-time ATTs, aggregated to an overall
    effect and an event-study path. Never-treated comparison group."""

    def fit(self, y, treat_start: Sequence) -> _DiDResult:
        mat = _as_matrix(y)
        return _DiDResult(_panelkit.fit_callaway_py(mat, _cohorts(treat_start, mat.shape[0])))


class SunAbraham:
    """Sun & Abraham (2021) interaction-weighted event study."""

    def fit(self, y, treat_start: Sequence) -> _DiDResult:
        mat = _as_matrix(y)
        return _DiDResult(_panelkit.fit_sunab_py(mat, _cohorts(treat_start, mat.shape[0])))


class _BaconResult:
    """Goodman-Bacon decomposition result."""

    def __init__(self, raw):
        self._raw = raw

    @property
    def twfe(self) -> float:
        """Weighted average of all 2x2 estimates — equals the TWFE coefficient."""
        return self._raw.twfe

    @property
    def forbidden_weight(self) -> float:
        """Total weight on forbidden (later-vs-earlier) comparisons."""
        return self._raw.forbidden_weight

    @property
    def components(self) -> list:
        """List of 2x2 comparisons (each with .kind, .treated_cohort,
        .comparison_cohort, .weight, .estimate)."""
        return list(self._raw.components)

    def summary(self) -> str:
        lines = [
            f"TWFE coefficient    : {self.twfe:.6g}",
            f"forbidden weight    : {self.forbidden_weight:.4f}"
            f"  (weight on already-treated-as-control comparisons)",
            "components:",
        ]
        for c in self.components:
            comp = "never" if c.comparison_cohort is None else f"g={c.comparison_cohort}"
            lines.append(
                f"  {c.kind:<28} g={c.treated_cohort} vs {comp:<7} "
                f"w={c.weight:7.4f}  beta={c.estimate:8.4f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return repr(self._raw)


class GoodmanBacon:
    """Goodman-Bacon (2021) decomposition of the TWFE DiD estimate into its
    constituent 2x2 comparisons and weights.

    A diagnostic, not an estimator: it shows how much of the TWFE coefficient
    rests on "forbidden" comparisons that use already-treated units as controls
    (the source of bias under heterogeneous effects). ``result.twfe`` reproduces
    the TWFE coefficient exactly.
    """

    def fit(self, y, treat_start: Sequence) -> _BaconResult:
        mat = _as_matrix(y)
        return _BaconResult(_panelkit.bacon_decompose_py(mat, _cohorts(treat_start, mat.shape[0])))
