"""Monte-Carlo comparison of the SC-family estimators and of ensembles built
from them — the evidence behind which methods belong in panelkit's geo ensemble.

    python benchmarks/sim_methods.py                 # all six DGPs (~20 s)
    python benchmarks/sim_methods.py --dgp hull_gap  # one scenario
    python benchmarks/sim_methods.py --mde           # + the power-engine view (~5 min)

What it measures, per data-generating process:

* **Bias / RMSE** of each estimator's ATT against the *known* true effect.
* **Error correlations** between estimators. This is the number that decides
  ensemble membership: averaging two estimators that make the same mistake buys
  nothing, averaging two that make different mistakes buys variance reduction.
* **Ensemble RMSE** for candidate member sets. Weights are estimated from the
  *null* runs (no injected effect) — exactly as the power engine estimates them
  from the historical-null windows — and only then applied to the effect runs,
  using only as many null runs as a real design has placebo windows
  (``--weight-reps``). Three weighting schemes are compared: the library's
  inverse-variance default, an equal-weight blend, and a correlation-aware
  minimum-variance blend.

With ``--mde`` it also runs the real geo power engine and compares each
ensemble's minimum detectable effect. The two views answer different questions
and it is worth knowing which one you are reading: see :func:`run_mde`.

The estimators are fit through ``_panelkit.fit_many``, which runs the whole
replication stack in parallel in Rust.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from panelkit import _panelkit

METHODS = ("SC", "ASC", "SDID", "FP", "RSC")

# Candidate ensembles: the shipped one, plus each addition on top of it.
CANDIDATE_ENSEMBLES = [
    ("SC", "ASC", "SDID"),            # what panelkit ships today
    ("SC", "ASC", "SDID", "FP"),      # + Ferman-Pinto demeaned SC
    ("SC", "ASC", "SDID", "RSC"),     # + robust / de-noised SC
    ("SC", "ASC", "SDID", "FP", "RSC"),
    ("SC", "SDID", "FP"),
    ("ASC", "SDID", "FP"),
    ("SDID", "FP"),
]


# ---------------------------------------------------------------------------
# Data-generating processes. Each returns the *untreated* panel Y(0), N x T.
# ---------------------------------------------------------------------------

def _ar1(rng, n, t, rho, sd):
    """n x t AR(1) noise with the stationary variance sd^2."""
    e = rng.normal(0.0, sd * np.sqrt(1.0 - rho**2), size=(n, t))
    out = np.empty((n, t))
    out[:, 0] = rng.normal(0.0, sd, size=n)
    for j in range(1, t):
        out[:, j] = rho * out[:, j - 1] + e[:, j]
    return out


def dgp_factor(rng, n, t, *, noise=1.0, hull="inside", n_factors=3):
    """Interactive-fixed-effects panel: Y = levels + loadings @ factors + noise.

    ``hull`` controls where unit 0 (the treated unit) sits relative to the donor
    pool: ``"inside"`` = an interior convex combination of donor loadings and
    levels (the case classic SC is built for); ``"outside"`` = loadings beyond
    the donor hull, so no convex combination can reproduce it.
    """
    f = np.cumsum(rng.normal(0, 1, size=(n_factors, t)), axis=1)  # smooth factors
    f += rng.normal(0, 0.5, size=(n_factors, t))
    load = rng.uniform(0.2, 1.2, size=(n, n_factors))
    level = rng.uniform(80.0, 120.0, size=n)
    if hull == "inside":
        # Treated unit is an interior mixture of a handful of donors.
        mix = rng.dirichlet(np.ones(5))
        load[0] = mix @ load[1:6]
        level[0] = mix @ level[1:6]
    else:
        load[0] = load[1:].max(axis=0) * 1.35
        level[0] = level[1:].max() * 1.30
    return level[:, None] + load @ f + rng.normal(0, noise, size=(n, t))


def dgp_hull_gap(rng, n, t):
    """Donors share a common factor structure; the treated unit's *level* sits
    far above every donor. Pure additive-fixed-effect mismatch — the case the
    Ferman-Pinto demeaned estimator is designed for."""
    f = np.cumsum(rng.normal(0, 1.0, size=(2, t)), axis=1)
    load = rng.uniform(0.5, 1.5, size=(n, 2))
    level = rng.uniform(90.0, 110.0, size=n)
    level[0] = 400.0                       # treated level way outside the hull
    return level[:, None] + load @ f + rng.normal(0, 1.0, size=(n, t))


def dgp_noisy(rng, n, t):
    """Low signal-to-noise: the pre-period fit cannot be good, so the weights are
    partly fitting noise (Ferman & Pinto's 'imperfect pre-treatment fit')."""
    return dgp_factor(rng, n, t, noise=6.0, hull="inside")


def dgp_geo(rng, n, t):
    """Realistic geo panel: heterogeneous market sizes, weekly seasonality,
    drifting trend, AR(1) noise — the shape of the data GeoDesign is used on."""
    size = np.exp(rng.normal(np.log(1000.0), 0.7, size=n))
    season = 1.0 + 0.25 * np.sin(2 * np.pi * np.arange(t) / 52.0)
    trend = 1.0 + np.cumsum(rng.normal(0.0, 0.004, size=(n, t)), axis=1)
    common = 1.0 + 0.08 * np.cumsum(rng.normal(0.0, 0.15, size=t)) / np.sqrt(t)
    base = size[:, None] * season[None, :] * trend * common[None, :]
    return base * (1.0 + _ar1(rng, n, t, rho=0.4, sd=0.05))


def dgp_lowrank_seasonal(rng, n, t):
    """Strongly low-rank with a dominant seasonal factor — the structure matrix
    completion / spectral methods are supposed to exploit."""
    per = 13
    f = np.vstack([
        np.sin(2 * np.pi * np.arange(t) / per),
        np.cos(2 * np.pi * np.arange(t) / per),
        np.linspace(0.0, 1.0, t),
    ])
    load = rng.uniform(0.5, 3.0, size=(n, 3)) * 20.0
    level = rng.uniform(200.0, 400.0, size=n)
    return level[:, None] + load @ f + rng.normal(0, 2.0, size=(n, t))


DGPS = {
    "factor_inside": lambda rng, n, t: dgp_factor(rng, n, t, hull="inside"),
    "factor_outside": lambda rng, n, t: dgp_factor(rng, n, t, hull="outside"),
    "hull_gap": dgp_hull_gap,
    "noisy": dgp_noisy,
    "geo": dgp_geo,
    "lowrank_seasonal": dgp_lowrank_seasonal,
}


# ---------------------------------------------------------------------------
# Monte-Carlo driver
# ---------------------------------------------------------------------------

def simulate(dgp_name, *, reps, n, t, t0, n_treated, lift, seed):
    """Build ``reps`` panels, apply ``lift`` to the treated block, fit every
    method, and return (errors, true_atts) with errors[m] an array of
    ``est - true`` per replication."""
    rng = np.random.default_rng(seed)
    make = DGPS[dgp_name]
    treated = list(range(n_treated))
    stack = np.empty((reps, n, t))
    true_att = np.empty(reps)
    for r in range(reps):
        y0 = make(rng, n, t)
        # True ATT = the mean post-period effect actually injected.
        true_att[r] = lift * y0[treated, t0:].mean()
        y = y0.copy()
        y[treated, t0:] *= 1.0 + lift
        stack[r] = y
    errs = {}
    for m in METHODS:
        est = np.asarray(_panelkit.fit_many(stack, treated, t0, m.lower()))
        errs[m] = est - true_att
    return errs, true_att


def inverse_variance_weights(null_errs, members, n=None):
    """The library's "auto" weighting: precision weights from the null spread."""
    var = np.array([np.var(_head(null_errs[m], n), ddof=1) for m in members])
    floor = 1e-6 * var.mean() + np.finfo(float).tiny
    prec = 1.0 / (var + floor)
    return prec / prec.sum()


def _head(x, n):
    return x if n is None else x[:n]


def min_variance_weights(null_errs, members, n=None, shrink=0.2):
    """Minimum-variance combination weights (Bates-Granger): the non-negative
    ``w`` summing to one that minimizes ``wᵀΣw``, where ``Σ`` is the covariance
    of the members' *null* errors.

    Inverse-variance weighting is only optimal when the members' errors are
    uncorrelated. They are not — SC and its demeaned cousin see the same shocks —
    so a member that adds variance without adding independent information gets
    over-weighted. This solves for the actual minimum-variance blend, with the
    covariance shrunk toward its diagonal to survive the small number of null
    windows a real design has. The simplex constraint is itself a regularizer.
    """
    x = np.column_stack([_head(null_errs[m], n) for m in members])
    sigma = np.cov(x, rowvar=False, ddof=1)
    sigma = np.atleast_2d(sigma)
    # Shrink toward the diagonal: (1-a)Σ + a·diag(Σ).
    sigma = (1.0 - shrink) * sigma + shrink * np.diag(np.diag(sigma))
    k = len(members)
    sigma = sigma + 1e-12 * np.trace(sigma) / k * np.eye(k)
    # Projected-gradient solve of min wᵀΣw over the simplex (small k, cheap).
    w = np.full(k, 1.0 / k)
    step = 1.0 / (np.abs(sigma).sum(axis=1).max() + 1e-300)
    for _ in range(5000):
        w_new = _project_simplex(w - step * (sigma @ w))
        if np.abs(w_new - w).max() < 1e-14:
            w = w_new
            break
        w = w_new
    return w


def _project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, len(v) + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


WEIGHTERS = {
    "invvar": inverse_variance_weights,
    "minvar": min_variance_weights,
    "equal": lambda null_errs, members, n=None: np.full(len(members), 1.0 / len(members)),
}


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def run_dgp(name, args):
    common = dict(reps=args.reps, n=args.n, t=args.t, t0=args.t0,
                  n_treated=args.n_treated, seed=args.seed)
    # Null run (no effect): the reference the weights and the critical value come
    # from. Effect run: the same panels with a real lift injected.
    null_errs, _ = simulate(name, lift=0.0, **common)
    eff_errs, true_att = simulate(name, lift=args.lift, **common)

    print(f"\n{'=' * 78}\nDGP: {name}   "
          f"(N={args.n}, T={args.t}, t0={args.t0}, treated={args.n_treated}, "
          f"lift={args.lift:.0%}, reps={args.reps})\n{'=' * 78}")
    scale = float(np.mean(np.abs(true_att))) or 1.0

    print(f"{'method':<8}{'bias':>12}{'RMSE':>12}{'RMSE %eff':>12}"
          f"{'null SD':>12}{'auto wt':>10}")
    base_w = inverse_variance_weights(null_errs, METHODS, args.weight_reps)
    for m, w in zip(METHODS, base_w):
        print(f"{m:<8}{np.mean(eff_errs[m]):>12.3f}{rmse(eff_errs[m]):>12.3f}"
              f"{100 * rmse(eff_errs[m]) / scale:>11.1f}%"
              f"{np.std(null_errs[m], ddof=1):>12.3f}{w:>10.2f}")

    print("\nerror correlations (effect run) — low = complementary in an ensemble")
    print(f"{'':<8}" + "".join(f"{m:>8}" for m in METHODS))
    for a in METHODS:
        row = "".join(f"{np.corrcoef(eff_errs[a], eff_errs[b])[0, 1]:>8.2f}"
                      for b in METHODS)
        print(f"{a:<8}{row}")

    schemes = list(WEIGHTERS)
    print(f"\nensemble RMSE — weights estimated from {args.weight_reps} null runs "
          f"(a real design has that many placebo windows)")
    print(f"{'members':<26}" + "".join(f"{s:>10}" for s in schemes)
          + f"{'best vs base':>15}")
    rows = {}
    for members in CANDIDATE_ENSEMBLES:
        cells = {}
        for scheme in schemes:
            w = WEIGHTERS[scheme](null_errs, members, args.weight_reps)
            cells[scheme] = rmse(sum(wi * eff_errs[m] for wi, m in zip(w, members)))
        rows[members] = cells
    base = rows[("SC", "ASC", "SDID")]
    for members, cells in rows.items():
        deltas = "".join(f"{cells[s]:>10.3f}" for s in schemes)
        best = min(cells[s] / base[s] - 1.0 for s in schemes)
        print(f"{'+'.join(members):<26}{deltas}{100 * best:>14.1f}%")
    return {m: rmse(eff_errs[m]) for m in METHODS}, rows


def run_mde(names, args):
    """The design-time view: run the *real* geo power engine (historical placebo
    with injected lift) on panels from each DGP and compare the ensembles' MDE.

    This measures something different from the Monte-Carlo above, and the two
    disagree in an informative way: the placebo design compares each estimator
    against its own historical-null spread, so a *constant* bias cancels out of
    the MDE entirely — while it lands squarely in the reported effect that
    ``evaluate()`` produces. An estimator can therefore look fine here and still
    be the reason a measured lift is wrong.
    """
    from panelkit import _panelkit

    lifts = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
    sets = {
        "SC+ASC+SDID": ["sc", "asc", "sdid"],
        "+FP": ["sc", "asc", "sdid", "fp"],
        "+RSC": ["sc", "asc", "sdid", "rsc"],
        "+FP+RSC": ["sc", "asc", "sdid", "fp", "rsc"],
    }
    print(f"\n{'=' * 78}\nMDE from the geo power engine — median over {args.panels} "
          f"panel draws\n{'=' * 78}")
    print(f"{'DGP':<18}" + "".join(f"{m.upper():>9}" for m in METHODS)
          + "  |" + "".join(f"{k:>14}" for k in sets))
    for name in names:
        rng = np.random.default_rng(args.seed)
        mdes = {k: [] for k in list(sets) + list(METHODS)}
        for _ in range(args.panels):
            y = np.ascontiguousarray(DGPS[name](rng, args.n, args.t))
            for m in METHODS:
                r = _panelkit.geo_power(y, [0], args.test_len, lifts, m.lower(),
                                        args.alpha, args.target_power, 0, None)
                mdes[m].append(r.mde_pct if r.mde_pct is not None else np.nan)
            for k, members in sets.items():
                r = _panelkit.geo_power_ensemble(
                    y, [0], args.test_len, lifts, args.alpha, args.target_power,
                    0, None, None, members)
                mdes[k].append(r.mde_pct if r.mde_pct is not None else np.nan)
        cell = lambda k: (f"{100 * np.nanmedian(mdes[k]):.2f}%"
                          if np.isfinite(mdes[k]).any() else "n/a")
        print(f"{name:<18}" + "".join(f"{cell(m):>9}" for m in METHODS)
              + "  |" + "".join(f"{cell(k):>14}" for k in sets))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dgp", default="all", choices=["all", *DGPS])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--n", type=int, default=40, help="units")
    ap.add_argument("--t", type=int, default=80, help="periods")
    ap.add_argument("--t0", type=int, default=60, help="first treated period")
    ap.add_argument("--n-treated", type=int, default=1)
    ap.add_argument("--lift", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--weight-reps", type=int, default=30,
                    help="null replications used to estimate ensemble weights "
                         "(mimics the number of historical placebo windows)")
    ap.add_argument("--scheme", default="invvar", choices=list(WEIGHTERS),
                    help="weighting scheme for the cross-DGP summary table")
    ap.add_argument("--mde", action="store_true",
                    help="also run the geo power engine and compare ensemble MDEs "
                         "(slower: it refits every sliding window)")
    ap.add_argument("--panels", type=int, default=8, help="panel draws for --mde")
    ap.add_argument("--test-len", type=int, default=8, help="test window for --mde")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--target-power", type=float, default=0.80)
    args = ap.parse_args()

    names = list(DGPS) if args.dgp == "all" else [args.dgp]
    t_start = time.time()
    summary = {}
    for name in names:
        summary[name] = run_dgp(name, args)

    if len(names) > 1:
        print(f"\n{'=' * 78}\nSUMMARY — ensemble RMSE vs the shipped SC+ASC+SDID"
              f"\n{'=' * 78}")
        print(f"(weighting scheme: {args.scheme})")
        print(f"{'members':<26}" + "".join(f"{n[:11]:>13}" for n in names))
        for members in CANDIDATE_ENSEMBLES:
            cells = []
            for name in names:
                rows = summary[name][1]
                base = rows[("SC", "ASC", "SDID")][args.scheme]
                cells.append(f"{100 * (rows[members][args.scheme] / base - 1):+12.1f}%")
            print(f"{'+'.join(members):<26}" + "".join(cells))

    if args.mde:
        run_mde(names, args)
    print(f"\n({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
