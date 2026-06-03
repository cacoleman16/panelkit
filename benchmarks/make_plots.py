"""Measure REAL timings (panelkit vs a NumPy + SciPy-SLSQP reference) and render
the performance figures embedded in the README.

Everything plotted here is measured live on the machine that runs this script —
no hard-coded numbers. Re-run to regenerate; absolute times depend on hardware.

Outputs:
  assets/bench_scaling.png  — per-fit time vs donor-pool size (log y)
  assets/bench_speedup.png  — single fit + full placebo wall time, with speedups
  assets/bench_results.txt  — the raw measured numbers
"""

import os
import platform
import statistics
import time

import numpy as np
from scipy.optimize import minimize

from panelkit import AugmentedSC, MCNNM, SyntheticControl, SyntheticDiD

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
os.makedirs(ASSETS, exist_ok=True)

T, T0 = 130, 104  # 104 pre + 26 post — a realistic geo panel
RANK = 3


def make_panel(n_units, seed=7):
    rng = np.random.default_rng(seed)
    uf = rng.normal(size=(n_units, RANK))
    tf = rng.normal(scale=0.5, size=(T, RANK))
    unit_level = 10.0 + rng.normal(size=n_units)
    time_level = np.cumsum(0.02 * rng.normal(size=T))
    Y = unit_level[:, None] + time_level[None, :] + uf @ tf.T
    Y[0, T0:] += 0.05
    return Y


def reference_sc(Y):
    """Classic Abadie SC via SciPy SLSQP on the simplex (the standard baseline)."""
    y_pre = Y[0, :T0]
    z0 = Y[1:, :T0].T
    j = z0.shape[1]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * j
    res = minimize(
        lambda w: float(np.sum((y_pre - z0 @ w) ** 2)),
        np.full(j, 1.0 / j),
        jac=lambda w: -2.0 * (z0.T @ (y_pre - z0 @ w)),
        bounds=bounds, constraints=cons, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x
    return float(np.mean(Y[0, T0:] - Y[1:, T0:].T @ w))


def _slsqp_sc_weights(z0, y_pre):
    j = z0.shape[1]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(
        lambda w: float(np.sum((y_pre - z0 @ w) ** 2)),
        np.full(j, 1.0 / j),
        jac=lambda w: -2.0 * (z0.T @ (y_pre - z0 @ w)),
        bounds=[(0.0, 1.0)] * j, constraints=cons, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x


def reference_asc(Y):
    """Augmented SC reference: SLSQP SC weights + closed-form ridge augmentation
    (matches panelkit's ASC math)."""
    y_pre = Y[0, :T0]
    z0 = Y[1:, :T0].T  # (T_pre, J)
    w = _slsqp_sc_weights(z0, y_pre)
    imbalance = y_pre - z0 @ w
    g = z0 @ z0.T  # (T_pre, T_pre)
    lam = 0.1 * np.trace(g) / g.shape[0]
    A = g + lam * np.eye(g.shape[0])
    donor_post = Y[1:, T0:].T  # (T_post, J)
    cf = []
    for tt in range(donor_post.shape[0]):
        dp = donor_post[tt]
        eta = np.linalg.solve(A, z0 @ dp)
        cf.append(dp @ w + imbalance @ eta)
    return float(np.mean(Y[0, T0:] - np.array(cf)))


def reference_sdid(Y):
    """Synthetic DiD reference (NumPy + SciPy SLSQP), mirroring panelkit's SDID:
    ridge-regularized unit weights + time weights (each a simplex QP with a
    concentrated-out intercept), then a doubly-weighted 2x2 DiD."""
    t_post = T - T0
    ytr = Y[0]
    ctrl = Y[1:]
    j = ctrl.shape[0]
    ctrl_pre = ctrl[:, :T0].T   # (T_pre, J)
    ctrl_post = ctrl[:, T0:].T  # (T_post, J)

    # Unit weights: match treated pre path, ridge zeta^2*T_pre, simplex+intercept.
    sd = np.diff(ctrl[:, :T0], axis=1).std(ddof=1)
    zeta = (1 * t_post) ** 0.25 * sd          # n_treated = 1
    eta_unit = zeta * zeta * T0
    col_mean = ctrl_pre.mean(axis=0)
    M = ctrl_pre - col_mean
    ytil = ytr[:T0] - ytr[:T0].mean()
    consu = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    omega = minimize(
        lambda w: float(np.sum((ytil - M @ w) ** 2) + eta_unit * (w @ w)),
        np.full(j, 1.0 / j),
        jac=lambda w: -2.0 * (M.T @ (ytil - M @ w)) + 2.0 * eta_unit * w,
        bounds=[(0.0, 1.0)] * j, constraints=consu, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    ).x

    # Time weights: match each control's post-avg from pre path, simplex+intercept.
    Dt = ctrl_pre.T  # (J, T_pre)
    cpost_avg = ctrl_post.mean(axis=0)  # (J,)
    Mt = Dt - Dt.mean(axis=0)
    ttil = cpost_avg - cpost_avg.mean()
    consl = [{"type": "eq", "fun": lambda l: np.sum(l) - 1.0}]
    lam = minimize(
        lambda l: float(np.sum((ttil - Mt @ l) ** 2)),
        np.full(T0, 1.0 / T0),
        jac=lambda l: -2.0 * (Mt.T @ (ttil - Mt @ l)),
        bounds=[(0.0, 1.0)] * T0, constraints=consl, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    ).x

    ytr_pre_lambda = float((lam * ytr[:T0]).sum())
    ytr_post = float(ytr[T0:].mean())
    ctrl_term = sum(
        omega[k] * (cpost_avg[k] - float((lam * ctrl_pre[:, k]).sum())) for k in range(j)
    )
    return (ytr_post - ytr_pre_lambda) - ctrl_term


def reference_mcnnm(Y, lam, max_iter=100, tol=1e-4):
    """MC-NNM SoftImpute using LAPACK SVD (np.linalg.svd) — the natural, highly
    optimized reference for the iterative-SVT inner loop."""
    n, t_ = Y.shape
    obs = np.ones((n, t_), dtype=bool)
    obs[0, T0:] = False
    L = np.zeros((n, t_))
    for _ in range(max_iter):
        M = np.where(obs, Y, L)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s2 = np.maximum(s - lam, 0.0)
        Lnew = (U * s2) @ Vt
        num = np.linalg.norm(Lnew - L)
        den = np.linalg.norm(L)
        L = Lnew
        if den > 0 and num / den < tol:
            break
    return float(np.mean(Y[0, T0:] - L[0, T0:]))


def median_time(fn, reps):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


# ---------------------------------------------------------------------------
# 1) Scaling sweep: per-fit time vs number of units (donor pool grows with N).
# ---------------------------------------------------------------------------
SIZES = [20, 50, 100, 150, 200]
# Report the MEDIAN over several panels (typical case). SciPy SLSQP has rare
# convergence cliffs on near-collinear donor panels (it can iterate to the cap),
# so the mean is dominated by outliers; the median is the fair representative.
SEEDS = [7, 13, 21, 42, 99]
pk_ms, ref_ms = [], []
model = SyntheticControl()
lines = [f"machine: {platform.platform()} | cpu: {platform.processor() or 'n/a'} | cores: {os.cpu_count()}"]
lines.append(f"panel: T={T} ({T0} pre / {T - T0} post); median over {len(SEEDS)} panels per size")
lines.append("")
lines.append(f"{'N_units':>8}{'panelkit_ms':>14}{'reference_ms':>14}{'speedup':>10}")
for n in SIZES:
    pk_runs, ref_runs = [], []
    for sd in SEEDS:
        Y = make_panel(n, seed=sd)
        pk_runs.append(median_time(lambda: model.fit(Y, treated=[0], treat_time=T0), reps=25))
        ref_runs.append(median_time(lambda: reference_sc(Y), reps=7))
    pk = statistics.median(pk_runs) * 1e3
    rf = statistics.median(ref_runs) * 1e3
    pk_ms.append(pk)
    ref_ms.append(rf)
    lines.append(f"{n:>8}{pk:>14.3f}{rf:>14.3f}{rf / pk:>9.1f}x")

# ---------------------------------------------------------------------------
# 2) Full placebo inference at N=200 (1 + 199 leave-one-out fits).
# ---------------------------------------------------------------------------
N_PB = 200
Ypb = make_panel(N_PB)
pb_model = SyntheticControl(inference="placebo")
t = time.perf_counter()
pb_model.fit(Ypb, treated=[0], treat_time=T0)  # warm
pk_pb = time.perf_counter() - t
t = time.perf_counter()
pb_model.fit(Ypb, treated=[0], treat_time=T0)
pk_pb = time.perf_counter() - t


def reference_placebo(Y):
    donors = np.arange(1, Y.shape[0])

    def ratio(yrow, pool):
        y_pre = yrow[:T0]
        z0 = pool[:, :T0].T
        j = z0.shape[1]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(
            lambda w: float(np.sum((y_pre - z0 @ w) ** 2)),
            np.full(j, 1.0 / j), jac=lambda w: -2.0 * (z0.T @ (y_pre - z0 @ w)),
            bounds=[(0.0, 1.0)] * j, constraints=cons, method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        w = res.x
        pre = np.sqrt(np.mean((y_pre - z0 @ w) ** 2))
        post = np.sqrt(np.mean((yrow[T0:] - pool[:, T0:].T @ w) ** 2))
        return post / pre if pre > 0 else np.inf

    tr = ratio(Y[0], Y[1:])
    pl = [ratio(Y[d], Y[[u for u in donors if u != d]]) for d in donors]
    return (1 + sum(r >= tr for r in pl)) / (1 + len(pl))


t = time.perf_counter()
reference_placebo(Ypb)
ref_pb = time.perf_counter() - t

# single-fit time at N=200 for the bar chart
pk_single = pk_ms[-1] / 1e3
ref_single = ref_ms[-1] / 1e3

lines.append("")
lines.append(f"full placebo @N={N_PB}: panelkit {pk_pb:.4f}s | reference {ref_pb:.3f}s "
             f"| speedup {ref_pb / pk_pb:.0f}x")

# ---------------------------------------------------------------------------
# 3) Per-fit time by method at N=200 (panelkit vs NumPy+SciPy reference).
#    SC / ASC / SDID are simplex-QP-based — Frank–Wolfe vs SLSQP. Median over
#    the same panels as the sweep.
# ---------------------------------------------------------------------------
asc_model = AugmentedSC()
sdid_model = SyntheticDiD()
pk_asc, ref_asc_ms, pk_sdid, ref_sdid_ms = [], [], [], []
for sd in SEEDS:
    Y = make_panel(N_PB, seed=sd)
    pk_asc.append(median_time(lambda: asc_model.fit(Y, treated=[0], treat_time=T0), reps=20))
    ref_asc_ms.append(median_time(lambda: reference_asc(Y), reps=7))
    pk_sdid.append(median_time(lambda: sdid_model.fit(Y, treated=[0], treat_time=T0), reps=20))
    ref_sdid_ms.append(median_time(lambda: reference_sdid(Y), reps=5))
methods = {
    "SC": (pk_single * 1e3, ref_single * 1e3),
    "ASC": (statistics.median(pk_asc) * 1e3, statistics.median(ref_asc_ms) * 1e3),
    "SDID": (statistics.median(pk_sdid) * 1e3, statistics.median(ref_sdid_ms) * 1e3),
}
lines.append("")
lines.append(f"per-fit time by method @N={N_PB} (ms):")
lines.append(f"{'method':>8}{'panelkit':>12}{'reference':>12}{'speedup':>10}")
for name, (p, r) in methods.items():
    lines.append(f"{name:>8}{p:>12.3f}{r:>12.3f}{r / p:>9.1f}x")

# Honest MC-NNM probe: panelkit's from-scratch Jacobi SVD vs LAPACK np.linalg.svd
# inside SoftImpute. We expect LAPACK to win — report it straight.
Ymc = make_panel(N_PB, seed=7)
smax = float(np.linalg.svd(Ymc, compute_uv=False)[0])
mc_lambda = 0.3 * smax
mc_model = MCNNM(lambda_=mc_lambda, max_iter=100, tol=1e-4)
pk_mc = median_time(lambda: mc_model.fit(Ymc, treated=[0], treat_time=T0), reps=3)
ref_mc = median_time(lambda: reference_mcnnm(Ymc, mc_lambda, 100, 1e-4), reps=3)
lines.append("")
lines.append(f"MC-NNM @N={N_PB} (fixed lambda, honest): panelkit {pk_mc * 1e3:.1f}ms | "
             f"LAPACK-SVD reference {ref_mc * 1e3:.1f}ms | ratio {ref_mc / pk_mc:.2f}x "
             f"({'panelkit faster' if ref_mc > pk_mc else 'reference faster — LAPACK SVD wins'})")

results = "\n".join(lines)
print(results)
with open(os.path.join(ASSETS, "bench_results.txt"), "w") as f:
    f.write(results + "\n")

# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PK = "#2563eb"
REF = "#9ca3af"

# Figure 1: scaling.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(SIZES, ref_ms, "o-", color=REF, label="NumPy + SciPy SLSQP")
ax.plot(SIZES, pk_ms, "o-", color=PK, label="panelkit (Rust)")
ax.set_yscale("log")
ax.set_xlabel("number of units (donor pool size)")
ax.set_ylabel("time per synthetic-control fit (ms, log scale)")
ax.set_title(f"Synthetic-control fit time vs panel size  (T={T})")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
for x, p, r in zip(SIZES, pk_ms, ref_ms):
    ax.annotate(f"{r / p:.0f}x", (x, p), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color=PK)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "bench_scaling.png"), dpi=150)

# Figure 2: speedup bars (single fit + full placebo), log scale.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
groups = ["single SC fit\n(N=200)", f"full placebo\n(N={N_PB}, {N_PB} fits)"]
pk_vals = [pk_single, pk_pb]
ref_vals = [ref_single, ref_pb]
x = np.arange(len(groups))
w = 0.38
ax.bar(x - w / 2, ref_vals, w, color=REF, label="NumPy + SciPy SLSQP")
ax.bar(x + w / 2, pk_vals, w, color=PK, label="panelkit (Rust)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title("panelkit vs reference — wall-clock time")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend()
for i, (pv, rv) in enumerate(zip(pk_vals, ref_vals)):
    ax.annotate(f"{rv / pv:.0f}x faster", (i + w / 2, pv), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9, color=PK, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "bench_speedup.png"), dpi=150)

# Figure 3: per-fit time by method (SC / ASC / SDID), panelkit vs reference.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
names = list(methods.keys())
pk_v = [methods[n][0] for n in names]
ref_v = [methods[n][1] for n in names]
x = np.arange(len(names))
w = 0.38
ax.bar(x - w / 2, ref_v, w, color=REF, label="NumPy + SciPy SLSQP")
ax.bar(x + w / 2, pk_v, w, color=PK, label="panelkit (Rust)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("time per fit (ms, log scale)")
ax.set_title(f"Per-fit time by estimator  (N={N_PB}, T={T})")
ax.grid(True, axis="y", which="both", alpha=0.25)
ax.legend()
for i, (pv, rv) in enumerate(zip(pk_v, ref_v)):
    ax.annotate(f"{rv / pv:.0f}x", (i + w / 2, pv), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9, color=PK, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "bench_methods.png"), dpi=150)

print("\nwrote assets/bench_scaling.png, bench_speedup.png, bench_methods.png, bench_results.txt")
