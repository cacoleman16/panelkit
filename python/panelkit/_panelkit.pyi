"""Type stubs for the compiled Rust extension `panelkit._panelkit`."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

class SCResult:
    att: float
    att_path: list[float]
    counterfactual: list[float]
    treated_post: list[float]
    weights: list[float]
    donor_ids: list[int]
    pre_rmspe: float
    post_rmspe: float
    p_value: Optional[float]
    se: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    inference_distribution: Optional[list[float]]
    placebo_atts: Optional[list[float]]
    def __repr__(self) -> str: ...

class CPASCResult:
    att: float
    p_value: float
    unit_ids: list[int]
    unit_att: list[float]
    unit_mspe: list[float]
    unit_weight: list[float]
    pooled_residual: list[float]
    null_residual: list[float]
    t0: int
    def __repr__(self) -> str: ...

class DiDResult:
    att: float
    se: float
    event_time: list[int]
    event_att: list[float]
    event_se: list[float]
    def __repr__(self) -> str: ...

class BaconComponent:
    kind: str
    treated_cohort: int
    comparison_cohort: Optional[int]
    weight: float
    estimate: float
    def __repr__(self) -> str: ...

class BaconResult:
    twfe: float
    forbidden_weight: float
    components: list[BaconComponent]
    def __repr__(self) -> str: ...

def version() -> str: ...
def fit_sc(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    treat_time: int,
    ridge: float = ...,
    placebo: bool = ...,
    level: float = ...,
) -> SCResult: ...
def fit_asc(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    treat_time: int,
    sc_ridge: float = ...,
    aug_lambda: Optional[float] = ...,
) -> SCResult: ...
def fit_sdid(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    treat_time: int,
    zeta_scale: float = ...,
) -> SCResult: ...
def fit_mcnnm(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    treat_time: int,
    lambda_: Optional[float] = ...,  # NOTE: matches the Rust binding's `lambda_`
    max_iter: int = ...,
    tol: float = ...,
    seed: int = ...,
    max_rank: Optional[int] = ...,
) -> SCResult: ...
def fit_cpasc(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    treat_time: int,
    mode: str = ...,
    n_strata: int = ...,
    block_len: Optional[int] = ...,
    sc_ridge: float = ...,
    aug_lambda: Optional[float] = ...,
) -> CPASCResult: ...
def bootstrap_mean(
    series: Sequence[float],
    kind: str = ...,
    block_len: int = ...,
    n_reps: int = ...,
    seed: int = ...,
    level: float = ...,
) -> tuple[float, float, float]: ...
def fit_many(
    y3: npt.NDArray[np.float64],
    treated: Sequence[int],
    treat_time: int,
    method: str = ...,
    ridge: float = ...,
    zeta_scale: float = ...,
) -> npt.NDArray[np.float64]: ...
def fit_twfe_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> DiDResult: ...
class PowerResult:
    method: str
    lifts: list[float]
    power: list[float]
    est_mean: list[float]
    est_lo: list[float]
    est_hi: list[float]
    mde_pct: Optional[float]
    mde_abs_per_period: Optional[float]
    mde_cumulative: Optional[float]
    crit: float
    se_null: float
    n_windows: int
    ensemble_weights: Optional[list[float]]
    def __repr__(self) -> str: ...

class GeoDiagnostics:
    holdout_pct: float
    pre_fit_rel: float
    improvement_vs_naive: float
    seasonality_strength: float
    stability_score: float
    confidence: float
    warnings: list[str]
    def __repr__(self) -> str: ...

class MarketCandidate:
    treated: list[int]
    power_at_target: float
    mde_pct: Optional[float]
    holdout_pct: float
    pre_fit_rel: float
    stability_score: float
    confidence: float
    score: float
    def __repr__(self) -> str: ...

def geo_power(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    test_len: int,
    lifts: Sequence[float],
    method: str = ...,
    alpha: float = ...,
    target_power: float = ...,
    min_pre: int = ...,
    lookback: Optional[int] = ...,
) -> PowerResult: ...
def geo_power_ensemble(
    y: npt.NDArray[np.float64],
    treated: Sequence[int],
    test_len: int,
    lifts: Sequence[float],
    alpha: float = ...,
    target_power: float = ...,
    min_pre: int = ...,
    lookback: Optional[int] = ...,
    weights: Optional[Sequence[float]] = ...,
) -> PowerResult: ...
def geo_diagnostics(
    y: npt.NDArray[np.float64], treated: Sequence[int], test_len: int
) -> GeoDiagnostics: ...
def geo_select(
    y: npt.NDArray[np.float64],
    eligible: Sequence[int],
    max_treated: int,
    test_len: int,
    target_lift: float,
    method: str = ...,
    alpha: float = ...,
    target_power: float = ...,
    min_pre: int = ...,
    n_candidates: int = ...,
    seed: int = ...,
    exact_size: Optional[int] = ...,
    lookback: Optional[int] = ...,
    include: Optional[Sequence[int]] = ...,
) -> list[MarketCandidate]: ...
def fit_callaway_py(
    y: npt.NDArray[np.float64],
    cohorts: Sequence[int],
    control: str = ...,
    covariates: Optional[npt.NDArray[np.float64]] = ...,
) -> DiDResult: ...
def fit_sunab_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> DiDResult: ...
def bacon_decompose_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> BaconResult: ...
