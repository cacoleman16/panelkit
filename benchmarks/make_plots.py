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

from panelkit import SyntheticControl

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

print("\nwrote assets/bench_scaling.png, assets/bench_speedup.png, assets/bench_results.txt")
