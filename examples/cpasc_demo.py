"""Demonstrates the novel CP-ASC family on a multi-treated-unit geo test.

CP-ASC fits one augmented synthetic control per treated unit, then pools the
per-unit effects with empirical-Bayes (inverse-MSPE) weighting — units that fit
their pre-period poorly are down-weighted — and tests the pooled effect by
conformal block permutation. Three pooling targets are available:

  * "mspe"       — CP-ASC: equal-ish, inverse-MSPE shrinkage pool.
  * "stratified" — Strat-CP-ASC: robust to a single extremal large unit.
  * "cumulative" — C-AS-CP-ASC: baseline-weighted (total-dollar) target.
"""

import numpy as np

from panelkit import CPASC

rng = np.random.default_rng(1)
n_treated, n_donor = 6, 8
N, T, T0 = n_treated + n_donor, 30, 22
TAU = 2.0

Y = np.zeros((N, T))
for u in range(n_treated, N):  # donors: random walks of differing size
    level = rng.normal()
    for t in range(T):
        level += 0.3 * rng.normal()
        Y[u, t] = 10.0 + level + 0.5 * u

for u in range(n_treated):  # treated: convex mix of two donors + effect
    a = 0.4 + 0.1 * u
    for t in range(T):
        Y[u, t] = a * Y[n_treated, t] + (1 - a) * Y[n_treated + 1, t]
        if t >= T0:
            Y[u, t] += TAU
        Y[u, t] += 0.05 * rng.normal()

treated = list(range(n_treated))

for mode in ("mspe", "stratified", "cumulative"):
    res = CPASC(mode=mode).fit(Y, treated, T0)
    print(f"[{mode:>10}] pooled ATT = {res.att:.4f}  conformal p = {res.p_value:.4f}")

print()
print(CPASC(mode="mspe").fit(Y, treated, T0).summary())
