"""Head-to-head speed benchmark: panelkit (Rust) vs a standard NumPy + SciPy
synthetic-control implementation, on a realistic geo panel (~200 units × 130
periods, the size the geocausal benchmark uses).

The reference is the textbook approach: minimize ||y_pre - Z0 w||^2 over the
simplex with scipy.optimize SLSQP. panelkit solves the same problem with its
from-scratch away-step Frank-Wolfe solver in Rust.

Run:  python benchmarks/bench_sc.py
"""

import time

import numpy as np
from scipy.optimize import minimize

from panelkit import SyntheticControl

# ---------------------------------------------------------------------------
# Panel: 200 units, 130 periods (104 pre + 26 post), 1 treated. Factor-model
# DGP with a known additive post-period effect.
# ---------------------------------------------------------------------------
N, T, T0 = 200, 130, 104
TAU = 0.05  # log-points, a realistic geo lift
rng = np.random.default_rng(7)

rank = 3
uf = rng.normal(size=(N, rank))
tf = rng.normal(scale=0.5, size=(T, rank))
unit_level = 10.0 + rng.normal(size=N)
time_level = np.cumsum(0.02 * rng.normal(size=T))
Y = unit_level[:, None] + time_level[None, :] + uf @ tf.T
Y[0, T0:] += TAU


def reference_sc(y, t0):
    """Classic Abadie SC via SciPy SLSQP on the simplex."""
    y_pre = y[0, :t0]
    z0 = y[1:, :t0].T  # (T_pre × J)
    j = z0.shape[1]
    w0 = np.full(j, 1.0 / j)

    def obj(w):
        r = y_pre - z0 @ w
        return float(r @ r)

    def grad(w):
        r = y_pre - z0 @ w
        return -2.0 * (z0.T @ r)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * j
    res = minimize(
        obj, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x
    cf_post = y[1:, t0:].T @ w
    att = float(np.mean(y[0, t0:] - cf_post))
    return att, w


def timeit(fn, reps):
    # warmup
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    elapsed = time.perf_counter() - t
    return elapsed / reps, out


REPS = 50
model = SyntheticControl()

pk_per, pk_out = timeit(lambda: model.fit(Y, treated=[0], treat_time=T0), REPS)
ref_per, ref_out = timeit(lambda: reference_sc(Y, T0), REPS)

pk_att = pk_out.att
ref_att = ref_out[0]

print(f"panel: {N} units × {T} periods ({T0} pre / {T-T0} post), {REPS} reps each")
print("-" * 64)
print(f"{'method':<28}{'ATT':>10}{'ms / fit':>14}")
print(f"{'panelkit (Rust FW)':<28}{pk_att:>10.5f}{pk_per*1e3:>14.3f}")
print(f"{'reference (NumPy+SLSQP)':<28}{ref_att:>10.5f}{ref_per*1e3:>14.3f}")
print("-" * 64)
print(f"speedup: {ref_per / pk_per:.1f}×   (true tau = {TAU})")
print(f"ATT agreement: |Δ| = {abs(pk_att - ref_att):.2e}")
