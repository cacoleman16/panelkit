"""Inference-level speed benchmark: a full SC placebo test (one fit for the
treated unit + one leave-one-out fit per donor) where the per-fit speedup
compounds and panelkit's parallelism kicks in.

panelkit runs the donor placebo fits multithreaded (rayon); the reference runs
the same number of SciPy-SLSQP fits in a Python loop.
"""

import time

import numpy as np
from scipy.optimize import minimize

from panelkit import SyntheticControl

N, T, T0 = 200, 130, 104
rng = np.random.default_rng(7)
rank = 3
uf = rng.normal(size=(N, rank))
tf = rng.normal(scale=0.5, size=(T, rank))
unit_level = 10.0 + rng.normal(size=N)
time_level = np.cumsum(0.02 * rng.normal(size=T))
Y = unit_level[:, None] + time_level[None, :] + uf @ tf.T
Y[0, T0:] += 0.05


def sc_ratio(y_row, donor_rows, t0):
    """RMSPE ratio for one (treated-or-placebo) unit via SciPy SLSQP."""
    y_pre = y_row[:t0]
    z0 = donor_rows[:, :t0].T
    j = z0.shape[1]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * j
    res = minimize(
        lambda w: float(np.sum((y_pre - z0 @ w) ** 2)),
        np.full(j, 1.0 / j),
        jac=lambda w: -2.0 * (z0.T @ (y_pre - z0 @ w)),
        bounds=bounds, constraints=cons, method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = res.x
    pre = np.sqrt(np.mean((y_pre - z0 @ w) ** 2))
    post = np.sqrt(np.mean((y_row[t0:] - donor_rows[:, t0:].T @ w) ** 2))
    return post / pre if pre > 0 else np.inf


def reference_placebo(y, t0):
    donors = np.arange(1, y.shape[0])  # treated = unit 0
    treated_ratio = sc_ratio(y[0], y[1:], t0)
    placebo = []
    for d in donors:
        pool = [u for u in donors if u != d]
        placebo.append(sc_ratio(y[d], y[pool], t0))
    n_extreme = sum(1 for r in placebo if r >= treated_ratio)
    return (1 + n_extreme) / (1 + len(placebo))


# panelkit (parallel placebo in Rust)
t = time.perf_counter()
res = SyntheticControl(inference="placebo").fit(Y, treated=[0], treat_time=T0)
pk_time = time.perf_counter() - t
pk_p = res.p_value

# reference (199 SLSQP fits in a Python loop)
t = time.perf_counter()
ref_p = reference_placebo(Y, T0)
ref_time = time.perf_counter() - t

print(f"panel: {N} units × {T} periods; placebo = 1 + {N-1} leave-one-out fits")
print("-" * 60)
print(f"{'method':<28}{'p-value':>10}{'seconds':>12}")
print(f"{'panelkit (Rust, parallel)':<28}{pk_p:>10.4f}{pk_time:>12.4f}")
print(f"{'reference (NumPy+SLSQP)':<28}{ref_p:>10.4f}{ref_time:>12.4f}")
print("-" * 60)
print(f"speedup: {ref_time / pk_time:.1f}×")
