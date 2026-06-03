"""Power analysis with panelkit's batched, parallel fitting.

Pattern: simulate many panels at each candidate effect size, fit them all in one
parallel Rust call (`fit_many`), and read the power curve off the resulting ATT
distributions. The whole loop stays in NumPy + one batched call per cell — no
Python per-fit loop, no multiprocessing.
"""

import time

import numpy as np

from panelkit import SyntheticControl

rng = np.random.default_rng(0)
N, T, T0 = 30, 40, 30
R = 2000                       # Monte-Carlo reps per effect size
EFFECTS = [0.0, 0.5, 1.0, 1.5, 2.0]
NOISE = 1.0


def simulate_stack(tau):
    """R panels: treated unit 0 = convex mix of two donors + tau post + noise."""
    stack = np.zeros((R, N, T))
    for r in range(R):
        for u in range(1, N):
            level = rng.normal()
            for t in range(T):
                level += 0.3 * rng.normal()
                stack[r, u, t] = 10 + level + u
        for t in range(T):
            base = 0.6 * stack[r, 1, t] + 0.4 * stack[r, 2, t]
            stack[r, 0, t] = base + (tau if t >= T0 else 0.0) + NOISE * rng.normal()
    return stack

model = SyntheticControl()

# First, a null run to set a rejection threshold (95th percentile of |ATT| under H0).
null_atts = np.abs(model.fit_many(simulate_stack(0.0), treated=[0], treat_time=T0))
crit = np.quantile(null_atts, 0.95)

print(f"{R} reps/cell, N={N}, T={T}; rejection threshold |ATT| > {crit:.3f}\n")
print(f"{'true tau':>9}{'mean ATT':>10}{'power':>9}")
t0 = time.perf_counter()
for tau in EFFECTS:
    atts = model.fit_many(simulate_stack(tau), treated=[0], treat_time=T0)
    power = float(np.mean(np.abs(atts) > crit))
    print(f"{tau:>9.2f}{atts.mean():>10.3f}{power:>9.2f}")
elapsed = time.perf_counter() - t0
total = R * (len(EFFECTS) + 1)
print(f"\n{total} fits in {elapsed:.2f}s  ({1e3 * elapsed / total:.3f} ms/fit, parallel)")
