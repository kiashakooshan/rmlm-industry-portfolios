"""baseline_order.py - independent baseline for the causal ORDER: causal tail coefficient (Gnecco et al. 2021; definition from
memory - verify against arXiv:1908.05097). Gamma_ij = mean over the k days where X_i is largest of U_j (rank-uniform of X_j).
Large Gamma_ij and small Gamma_ji suggests i -> j. Node score = mean_j (Gamma_ij - Gamma_ji); sources get the highest score.
Usage (from python/):  python baseline_order.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15
Prints: top/bottom nodes, Kendall tau between this ordering and the RMLM Algorithm-1 generations."""
import argparse, numpy as np, pandas as pd
from scipy.stats import kendalltau
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *
def tail_coef_scores(X, kfrac=0.05):
    n, d = X.shape; U = (np.argsort(np.argsort(X, axis=0), axis=0) + 1) / (n + 1); k = max(int(kfrac * n), 5)
    G = np.zeros((d, d))
    for i in range(d):
        idx = np.argsort(-X[:, i])[:k]; G[i] = U[idx].mean(axis=0)
    np.fill_diagonal(G, 0); return (G - G.T).sum(axis=1) / (d - 1)
def gen_rank(groups, d):
    r = np.zeros(d)
    for g, nodes in enumerate(groups):
        for v in nodes: r[v] = g
    return r
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end"); a = p.parse_args()
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; X = df.values.astype(float); names = list(df.columns)
    sc = tail_coef_scores(X); Xf = to_frechet(X); g = algorithm1(Xf, 1.3, 0.1, int(.11 * len(X))); r = gen_rank(g, len(names))
    print("causal-tail-coefficient ordering (sources first):", [names[i] for i in np.argsort(-sc)])
    print("Kendall tau (tail-coef score vs RMLM generation; negative = they agree):", round(kendalltau(sc, r)[0], 3))
