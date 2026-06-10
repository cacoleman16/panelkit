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
        self._boot_se = None
        self._boot_ci = None
        self._level = 0.95

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
        # Bootstrap-derived SE overrides the placebo-derived one when present.
        return self._boot_se if self._boot_se is not None else self._raw.se

    @property
    def ci(self):
        """(lower, upper) confidence interval, when an inference engine ran."""
        return self._boot_ci

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
        if self.se is not None:
            lines.append(f"SE             : {self.se:.6g}")
        if self.ci is not None:
            lines.append(f"{int(100*self._level)}% CI         : [{self.ci[0]:.6g}, {self.ci[1]:.6g}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return repr(self._raw)


def _as_stack(panels) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(panels, dtype=np.float64))
    if arr.ndim != 3:
        raise ValueError(
            f"panel stack must be 3-D (R reps × N units × T periods), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "panel stack contains NaN or inf; panelkit requires complete, finite panels"
        )
    return arr


def _as_index_list(name, values) -> list:
    """Coerce unit indices / period markers to ints, rejecting bools and
    non-integral floats (``treated=[0.9]`` silently truncating to unit 0 is a
    bug factory)."""
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integer indices, got a bool ({v!r})")
        f = float(v)
        if not f.is_integer():
            raise ValueError(f"{name} must contain integer indices, got {v!r}")
        out.append(int(f))
    return out


def _as_period(name, value) -> int:
    """Coerce a period index to int, rejecting bools and non-integral floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer period index, got a bool ({value!r})")
    f = float(value)
    if not f.is_integer():
        raise ValueError(f"{name} must be an integer period index, got {value!r}")
    return int(f)


def _check_inference(inference, supported, cls_name):
    """Reject unknown inference engines up front — a typo must not silently
    skip inference."""
    if inference is not None and inference not in supported:
        opts = ", ".join(repr(s) for s in supported)
        raise ValueError(
            f"{cls_name} does not support inference={inference!r}; "
            f"choose one of None, {opts}"
        )


def _check_level(level: float) -> float:
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level}")
    return level


def _attach_bootstrap(result, inference, level, block_len, n_reps, seed):
    """Run a block/stationary bootstrap of the post-period gap and attach
    se / ci to an SC-family result."""
    result._level = level
    if inference in ("block", "stationary"):
        se, lo, hi = _panelkit.bootstrap_mean(
            list(result.att_path), inference, int(block_len), int(n_reps), int(seed), level
        )
        result._boot_se = se
        result._boot_ci = (lo, hi)
    return result


def _as_matrix(y) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"y must be 2-D (N units × T periods), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y contains NaN or inf; panelkit requires a complete, finite panel")
    return arr


def _validate_block(mat: np.ndarray, treated, treat_time: int) -> None:
    """Validate block-treatment args against the panel shape, raising ValueError
    (rather than letting the Rust core panic) on bad input."""
    n, t = mat.shape
    if not treated:
        raise ValueError("`treated` must list at least one treated unit index")
    for u in treated:
        if not (0 <= int(u) < n):
            raise ValueError(f"treated unit index {u} out of range [0, {n})")
    if not (1 <= int(treat_time) < t):
        raise ValueError(
            f"treat_time {treat_time} must be in [1, {t}) so there is ≥1 pre and ≥1 post period"
        )
    if len(treated) >= n:
        raise ValueError("need at least one never-treated unit as a donor/control")


class SyntheticControl:
    """Synthetic Control (Abadie, Diamond & Hainmueller 2010).

    Parameters
    ----------
    ridge:
        Ridge penalty on the donor weights (0.0 = classic SC).
    inference:
        ``"placebo"`` (in-space placebo p-value), ``"block"`` / ``"stationary"``
        (block / stationary bootstrap of the post-period gap → SE + CI), or
        ``None``.
    level:
        Confidence level for reported intervals.
    block_len, n_reps, seed:
        Bootstrap settings used when ``inference`` is ``"block"``/``"stationary"``.
    """

    def __init__(
        self,
        ridge: float = 0.0,
        inference: str | None = None,
        level: float = 0.95,
        block_len: int = 4,
        n_reps: int = 2000,
        seed: int = 0,
    ):
        _check_inference(inference, ("placebo", "block", "stationary"), "SyntheticControl")
        self.ridge = ridge
        self.inference = inference
        self.level = _check_level(level)
        self.block_len = block_len
        self.n_reps = n_reps
        self.seed = seed

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(mat, treated, treat_time)
        do_placebo = self.inference == "placebo"
        raw = _panelkit.fit_sc(
            mat,
            treated,
            treat_time,
            self.ridge,
            do_placebo,
            self.level,
        )
        return _attach_bootstrap(
            _Result(raw), self.inference, self.level, self.block_len, self.n_reps, self.seed
        )

    def fit_many(self, panels, treated: Sequence[int], treat_time: int) -> np.ndarray:
        """Fit across a stack of panels in parallel (for Monte-Carlo / power /
        robustness runs). ``panels`` is ``(R, N, T)``; returns ``R`` ATTs."""
        stack = _as_stack(panels)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(np.zeros(stack.shape[1:]), treated, treat_time)
        return np.asarray(
            _panelkit.fit_many(stack, treated, treat_time, "sc", self.ridge, 1.0),
            dtype=float,
        )


class AugmentedSC:
    """Augmented Synthetic Control (Ben-Michael, Feller & Rothstein 2021).

    Corrects residual pre-treatment imbalance with a ridge outcome model.
    """

    def __init__(
        self,
        sc_ridge: float = 0.0,
        aug_lambda: float | None = None,
        inference: str | None = None,
        level: float = 0.95,
        block_len: int = 4,
        n_reps: int = 2000,
        seed: int = 0,
    ):
        _check_inference(inference, ("block", "stationary"), "AugmentedSC")
        if aug_lambda is not None and not (float(aug_lambda) > 0.0):
            raise ValueError(
                f"aug_lambda must be > 0 (or None for the automatic choice); got {aug_lambda}"
            )
        self.sc_ridge = sc_ridge
        self.aug_lambda = aug_lambda
        self.inference = inference
        self.level = _check_level(level)
        self.block_len = block_len
        self.n_reps = n_reps
        self.seed = seed

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(mat, treated, treat_time)
        raw = _panelkit.fit_asc(
            mat, treated, treat_time, self.sc_ridge, self.aug_lambda
        )
        return _attach_bootstrap(
            _Result(raw), self.inference, self.level, self.block_len, self.n_reps, self.seed
        )

    def fit_many(self, panels, treated: Sequence[int], treat_time: int) -> np.ndarray:
        """Fit across a stack of panels ``(R, N, T)`` in parallel; returns R ATTs."""
        stack = _as_stack(panels)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(np.zeros(stack.shape[1:]), treated, treat_time)
        return np.asarray(
            _panelkit.fit_many(stack, treated, treat_time, "asc", self.sc_ridge, 1.0),
            dtype=float,
        )


class SyntheticDiD:
    """Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).

    The recommended general-purpose default: unit + time weights feeding a
    doubly-weighted 2×2 difference-in-differences. Pass ``inference="block"`` or
    ``"stationary"`` for a bootstrap SE + CI on the ATT.
    """

    def __init__(
        self,
        zeta_scale: float = 1.0,
        inference: str | None = None,
        level: float = 0.95,
        block_len: int = 4,
        n_reps: int = 2000,
        seed: int = 0,
    ):
        _check_inference(inference, ("block", "stationary"), "SyntheticDiD")
        self.zeta_scale = zeta_scale
        self.inference = inference
        self.level = _check_level(level)
        self.block_len = block_len
        self.n_reps = n_reps
        self.seed = seed

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(mat, treated, treat_time)
        raw = _panelkit.fit_sdid(mat, treated, treat_time, self.zeta_scale)
        return _attach_bootstrap(
            _Result(raw), self.inference, self.level, self.block_len, self.n_reps, self.seed
        )

    def fit_many(self, panels, treated: Sequence[int], treat_time: int) -> np.ndarray:
        """Fit across a stack of panels ``(R, N, T)`` in parallel; returns R ATTs."""
        stack = _as_stack(panels)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(np.zeros(stack.shape[1:]), treated, treat_time)
        return np.asarray(
            _panelkit.fit_many(stack, treated, treat_time, "sdid", 0.0, self.zeta_scale),
            dtype=float,
        )


class MCNNM:
    """Matrix-Completion NNM (Athey et al. 2021).

    Estimates a low-rank counterfactual by iterative singular-value
    thresholding (SoftImpute). ``lambda_`` is chosen by cross-validation when
    left as ``None``.

    ``max_rank`` (optional) switches the inner SVD to a fast **randomized
    truncated SVD** of that rank — a large speedup when the counterfactual is
    low-rank, while staying dependency-free. Leave ``None`` for an exact SVD.
    """

    def __init__(
        self,
        lambda_: float | None = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        seed: int = 0,
        max_rank: int | None = None,
    ):
        if lambda_ is not None and not (float(lambda_) > 0.0):
            raise ValueError(
                f"lambda_ must be > 0 (or None for cross-validation); got {lambda_}"
            )
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.max_rank = max_rank

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _Result:
        mat = _as_matrix(y)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(mat, treated, treat_time)
        raw = _panelkit.fit_mcnnm(
            mat,
            treated,
            treat_time,
            self.lambda_,
            int(self.max_iter),
            float(self.tol),
            int(self.seed),
            None if self.max_rank is None else int(self.max_rank),
        )
        return _Result(raw)


class _CPASCResult:
    """Result of a CP-ASC-family fit: pooled ATT, per-unit detail, conformal p."""

    def __init__(self, raw):
        self._raw = raw

    @property
    def att(self) -> float:
        return self._raw.att

    @property
    def p_value(self) -> float:
        """Conformal block-permutation p-value."""
        return self._raw.p_value

    @property
    def unit_ids(self) -> np.ndarray:
        return np.asarray(self._raw.unit_ids, dtype=int)

    @property
    def unit_att(self) -> np.ndarray:
        return np.asarray(self._raw.unit_att, dtype=float)

    @property
    def unit_mspe(self) -> np.ndarray:
        return np.asarray(self._raw.unit_mspe, dtype=float)

    @property
    def unit_weight(self) -> np.ndarray:
        return np.asarray(self._raw.unit_weight, dtype=float)

    @property
    def pooled_residual(self) -> np.ndarray:
        """Pooled residual path of the main fit (descriptive)."""
        return np.asarray(self._raw.pooled_residual, dtype=float)

    @property
    def null_residual(self) -> np.ndarray:
        """Pooled residual path of the null-imposed full-sample refit — the
        exchangeable path the conformal permutation actually tests."""
        return np.asarray(self._raw.null_residual, dtype=float)

    def summary(self) -> str:
        lines = [
            f"pooled ATT      : {self.att:.6g}",
            f"conformal p     : {self.p_value:.4g}",
            f"# treated units : {len(self.unit_ids)}",
            "per-unit (id: att, weight, mspe):",
        ]
        for i, a, w, m in zip(self.unit_ids, self.unit_att, self.unit_weight, self.unit_mspe):
            lines.append(f"  {int(i):>3d}: att={a:8.4f}  w={w:6.4f}  mspe={m:.4g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return repr(self._raw)


class CPASC:
    """Conformal Pooled Augmented Synthetic Control family (novel).

    Fits one augmented SC per treated unit, then pools the per-unit effects with
    empirical-Bayes shrinkage and tests the pooled effect by conformal block
    permutation. Modes:

    - ``"mspe"`` (CP-ASC): inverse-MSPE empirical-Bayes pooling.
    - ``"stratified"`` (Strat-CP-ASC): stratify by size into ``n_strata`` bins,
      pool within each, average — robust to a single extremal large unit.
    - ``"cumulative"`` (C-AS-CP-ASC): baseline-size-weighted target (maps to
      total/cumulative dollar lift rather than the equal-weighted average).
    """

    def __init__(
        self,
        mode: str = "mspe",
        n_strata: int = 3,
        block_len: int | None = None,
        sc_ridge: float = 0.0,
        aug_lambda: float | None = None,
    ):
        if mode not in ("mspe", "stratified", "cumulative"):
            raise ValueError(
                f"unknown CPASC mode {mode!r}; choose 'mspe', 'stratified', or 'cumulative'"
            )
        if aug_lambda is not None and not (float(aug_lambda) > 0.0):
            raise ValueError(
                f"aug_lambda must be > 0 (or None for the automatic choice); got {aug_lambda}"
            )
        self.mode = mode
        self.n_strata = n_strata
        self.block_len = block_len
        self.sc_ridge = sc_ridge
        self.aug_lambda = aug_lambda

    def fit(self, y, treated: Sequence[int], treat_time: int) -> _CPASCResult:
        mat = _as_matrix(y)
        treated = _as_index_list("treated", treated)
        treat_time = _as_period("treat_time", treat_time)
        _validate_block(mat, treated, treat_time)
        raw = _panelkit.fit_cpasc(
            mat,
            treated,
            treat_time,
            self.mode,
            int(self.n_strata),
            self.block_len,
            self.sc_ridge,
            self.aug_lambda,
        )
        return _CPASCResult(raw)


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
    """Normalize a per-unit treatment-start spec to int cohorts (<0 = never).

    A cohort at or beyond the last period means the unit is never treated
    *within the sample*; the core normalizes it to never-treated (the R ``did``
    package convention). Bools and non-integral floats are rejected.
    """
    out = []
    for c in treat_start:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            out.append(-1)
        elif isinstance(c, (bool, np.bool_)):
            raise ValueError(f"treat_start must contain integer periods, got a bool ({c!r})")
        else:
            f = float(c)
            if not f.is_integer():
                raise ValueError(f"treat_start must contain integer periods, got {c!r}")
            out.append(int(f))
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
    effect and an event-study path.

    Parameters
    ----------
    control_group:
        ``"never"`` (never-treated, default) or ``"notyet"`` (not-yet-treated —
        a larger control pool that also works without never-treated units).
    """

    def __init__(self, control_group: str = "never"):
        if control_group not in ("never", "notyet", "not_yet_treated"):
            raise ValueError(
                f"control_group must be 'never' or 'notyet'; got {control_group!r}"
            )
        self.control_group = control_group

    def fit(self, y, treat_start: Sequence, covariates=None) -> _DiDResult:
        """Fit C&S. Pass ``covariates`` (an ``N×K`` array of time-invariant unit
        characteristics) for covariate-adjusted (regression-adjustment) ATTs."""
        mat = _as_matrix(y)
        cov = None
        if covariates is not None:
            cov = np.ascontiguousarray(np.asarray(covariates, dtype=np.float64))
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            if cov.ndim != 2 or cov.shape[0] != mat.shape[0]:
                raise ValueError(
                    f"covariates must be (N, K) with N={mat.shape[0]} rows, got {cov.shape}"
                )
        return _DiDResult(
            _panelkit.fit_callaway_py(
                mat, _cohorts(treat_start, mat.shape[0]), self.control_group, cov
            )
        )


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
