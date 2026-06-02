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
    def __repr__(self) -> str: ...

class CPASCResult:
    att: float
    p_value: float
    unit_ids: list[int]
    unit_att: list[float]
    unit_mspe: list[float]
    unit_weight: list[float]
    pooled_residual: list[float]
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
    lambda_: Optional[float] = ...,
    max_iter: int = ...,
    tol: float = ...,
    seed: int = ...,
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
def fit_twfe_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> DiDResult: ...
def fit_callaway_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> DiDResult: ...
def fit_sunab_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> DiDResult: ...
def bacon_decompose_py(y: npt.NDArray[np.float64], cohorts: Sequence[int]) -> BaconResult: ...
