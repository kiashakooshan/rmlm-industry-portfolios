"""sample_size_curve.py - professor: 'use larger windows, then add sample size'. How does stability change with the window length L?
For each L (years) all windows of length L (step 1 year) are analysed: pipeline on the window, plus B block-bootstrap resamples of the window
(FULL = order re-estimated, FIXED = window's own order kept). Reported per L (mean over windows):
  stable edges (freq>=0.7) FULL / FIXED, mean Kendall tau of bootstrap order vs window order, nSHD to the previous window of the same length,
  nSHD to the full-sample graph, number of edges.
Usage (from python/):  python sample_size_curve.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 30
Outputs -> ../results/sample_size/ : sample_size_curve.csv, sample_size_windows.csv, fig_sample_size.png
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def analyse(Xw, B, block, delta, rng):
    n, d = Xw.shape; ko, kA = max(20, int(.11 * n)), max(15, int(.04 * n))
    e0, A0, g0 = pipeline(Xw, 1.3, 0.1, ko, kA, delta); r0 = rank_of(g0, d); cf, cx, taus = {}, {}, []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, int(np.ceil(n / block))); Xb = np.concatenate([Xw[s:s + block] for s in st])[:n]
        e, _, g = pipeline(Xb, 1.3, 0.1, ko, kA, delta)
        for x in e: cf[x] = cf.get(x, 0) + 1
        taus.append(kendalltau(rank_of(g, d), r0)[0])
        for x in hard_threshold_dag(nrm(algorithm2(to_frechet(Xb), g0, kA)), delta): cx[x] = cx.get(x, 0) + 1
    cnt = lambda c, t: sum(v / B >= t for v in c.values())
    return e0, dict(n_edges=len(e0), full07=cnt(cf, .7), fixed07=cnt(cx, .7), full05=cnt(cf, .5), fixed05=cnt(cx, .5), tau=float(np.nanmean(taus)))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--boot", type=int, default=30); p.add_argument("--block", type=int, default=20); p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--lengths", type=float, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 9]); p.add_argument("--out", default="../results/sample_size"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; X = df.values.astype(float); n = len(X); S = 252
    e_full, _ = analyse(X, a.boot, a.block, a.delta, rng); rows = []
    for L in a.lengths:
        W = int(L * 252); prev = None
        for s0 in range(0, n - W + 1, S):
            e, m = analyse(X[s0:s0 + W], a.boot, a.block, a.delta, rng)
            m.update(L=L, window=f"{df.index[s0].year}-{df.index[min(s0+W, n)-1].year}", nshd_prev=nshd(e, prev) if prev is not None else np.nan, nshd_full=nshd(e, e_full)); rows.append(m); prev = e
        print("done L =", L, flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{a.out}/sample_size_windows.csv", index=False)
    G = R.groupby("L").agg(windows=("window", "count"), n_edges=("n_edges", "mean"), stable07_full=("full07", "mean"), stable07_fixed=("fixed07", "mean"), stable05_full=("full05", "mean"), stable05_fixed=("fixed05", "mean"), tau=("tau", "mean"), nshd_prev=("nshd_prev", "mean"), nshd_full=("nshd_full", "mean")).reset_index()
    G.to_csv(f"{a.out}/sample_size_curve.csv", index=False); print(G.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].plot(G.L, G.stable07_full, "o-", label="full"); ax[0].plot(G.L, G.stable07_fixed, "s--", label="fixed order"); ax[0].set_title("edges with bootstrap freq >= 0.7", fontsize=9); ax[0].legend(fontsize=8)
    ax[1].plot(G.L, G.tau, "o-"); ax[1].set_title("Kendall tau: bootstrap order vs window order", fontsize=9)
    ax[2].plot(G.L, G.nshd_prev, "o-"); ax[2].set_title("nSHD to previous window", fontsize=9)
    ax[3].plot(G.L, G.nshd_full, "o-"); ax[3].set_title("nSHD to the full-sample graph", fontsize=9)
    for x in ax: x.set_xlabel("window length (years)")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_sample_size.png", dpi=200); plt.close(); print("saved to", a.out)

if __name__ == "__main__":
    main()
