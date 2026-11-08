"""boot_fixed_order.py - where does the instability come from: the ORDER (Algorithm 1) or the COEFFICIENTS (Algorithm 2 + threshold)?
Block bootstrap, two variants:  FULL = whole pipeline re-run (as in the paper) ; FIXED = causal order from the original data kept fixed.
Prints: #edges with freq>=0.5/0.7/0.9 for both, how often each node is in generation 0 (FULL), and mean Kendall tau of the
bootstrap generation ranks against the original ranks.
Usage (from python/):  python boot_fixed_order.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100
"""
import argparse, numpy as np, pandas as pd
from scipy.stats import kendalltau
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *
def nrm(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
def rank_of(groups, d):
    r = np.zeros(d)
    for g, nodes in enumerate(groups):
        for v in nodes: r[v] = g
    return r
def run_both(X, B=100, block=20, seed=0, names=None, delta=0.1):
    rng = np.random.default_rng(seed); n, d = X.shape; kA = int(.04 * n); ko = int(.11 * n)
    Xf = to_frechet(X); g0 = algorithm1(Xf, 1.3, 0.1, ko); r0 = rank_of(g0, d)
    cf, cx, gen0, taus = {}, {}, np.zeros(d), []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, int(np.ceil(n / block))); Xb = np.concatenate([X[s:s + block] for s in st])[:n]
        e, _, g = pipeline(Xb, 1.3, 0.1, ko, kA, delta)
        for x in e: cf[x] = cf.get(x, 0) + 1
        for v in g[0]: gen0[v] += 1
        taus.append(kendalltau(rank_of(g, d), r0)[0])
        A = nrm(algorithm2(to_frechet(Xb), g0, kA))
        for x in hard_threshold_dag(A, delta): cx[x] = cx.get(x, 0) + 1
    cnt = lambda c, t: sum(v / B >= t for v in c.values())
    return dict(full=[cnt(cf, t) for t in (.5, .7, .9)], fixed=[cnt(cx, t) for t in (.5, .7, .9)], gen0=gen0 / B, tau=float(np.nanmean(taus)))
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--boot", type=int, default=100)
    a = p.parse_args(); df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names = list(df.columns)
    r = run_both(df.values.astype(float), B=a.boot)
    print("edges with freq >= .5/.7/.9  FULL (order re-estimated):", r["full"], "  FIXED order:", r["fixed"])
    print("mean Kendall tau of bootstrap generations vs original:", round(r["tau"], 3))
    top = np.argsort(-r["gen0"])[:6]; print("most frequent generation-0 nodes:", [(names[i], round(float(r["gen0"][i]), 2)) for i in top])
