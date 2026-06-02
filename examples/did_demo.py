"""Demonstrates the headline modern-DiD result: under staggered adoption with
heterogeneous effects, TWFE is biased (it makes "forbidden comparisons" using
already-treated units as controls), while Callaway-Sant'Anna and Sun-Abraham
recover the true average treatment effect.
"""

import numpy as np

from panelkit import TWFE, CallawaySantAnna, SunAbraham

rng = np.random.default_rng(4)

per_group, T = 25, 16
N = per_group * 3
g1, g2 = 5, 10            # early / late adoption periods
eff_early, eff_late = 1.0, 8.0   # strongly heterogeneous effects

unit_fe = 3.0 + rng.normal(size=N)
time_fe = np.cumsum(0.05 * rng.normal(size=T))

treat_start = []
Y = np.zeros((N, T))
eff_sum = eff_cnt = 0.0
for i in range(N):
    grp = i // per_group  # 0 never, 1 early, 2 late
    start, eff = {0: (None, 0.0), 1: (g1, eff_early), 2: (g2, eff_late)}[grp]
    treat_start.append(-1 if start is None else start)
    for t in range(T):
        v = unit_fe[i] + time_fe[t] + 0.1 * rng.normal()
        if start is not None and t >= start:
            v += eff
            eff_sum += eff
            eff_cnt += 1
        Y[i, t] = v

true_att = eff_sum / eff_cnt

twfe = TWFE().fit(Y, treat_start)
cs = CallawaySantAnna().fit(Y, treat_start)
sa = SunAbraham().fit(Y, treat_start)

print(f"true average ATT on treated : {true_att:.3f}\n")
print(f"{'estimator':<22}{'ATT':>9}{'error':>9}")
print("-" * 40)
print(f"{'TWFE (biased)':<22}{twfe.att:>9.3f}{abs(twfe.att-true_att):>9.3f}")
print(f"{'Callaway-SantAnna':<22}{cs.att:>9.3f}{abs(cs.att-true_att):>9.3f}")
print(f"{'Sun-Abraham':<22}{sa.att:>9.3f}{abs(sa.att-true_att):>9.3f}")
print()
print("C&S event study (pre-trends near 0, post = effects):")
print(cs.summary())
