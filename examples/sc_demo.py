"""End-to-end smoke test / demo of the synthetic-control vertical slice.

Builds a panel where the treated unit is, pre-treatment, an exact convex mix of
two donors; a known effect tau is added post-treatment. panelkit should recover
tau and place weight on the two real donors.
"""

import numpy as np

from panelkit import SyntheticControl

rng = np.random.default_rng(0)

N, T, T0 = 8, 30, 22
TAU = 3.0

y = np.zeros((N, T))
# Donors 1..N-1 are smooth random walks.
for u in range(1, N):
    level = rng.normal()
    for t in range(T):
        level += 0.3 * rng.normal()
        y[u, t] = 10.0 + level + 0.5 * u

# Treated unit 0 = 0.6*donor1 + 0.4*donor2, plus TAU in the post-period.
for t in range(T):
    base = 0.6 * y[1, t] + 0.4 * y[2, t]
    y[0, t] = base + (TAU if t >= T0 else 0.0)

model = SyntheticControl(inference="placebo")
res = model.fit(y, treated=[0], treat_time=T0)

print(res.summary())
print("repr:", repr(res))
print(f"\nrecovered ATT = {res.att:.4f}  (true tau = {TAU})")
print("weight on donor 1:", round(float(res.weights[res.donor_ids.tolist().index(1)]), 4))
print("weight on donor 2:", round(float(res.weights[res.donor_ids.tolist().index(2)]), 4))
print("att_path:", np.round(res.att_path, 3))

assert abs(res.att - TAU) < 1e-2, "ATT should recover tau"
assert res.p_value is not None
print("\nOK: end-to-end SC slice works.")
