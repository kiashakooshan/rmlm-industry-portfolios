"""perm_edges.py - permutation test per edge (+ multiple-testing correction) for the RMLM graph.
Idea (professor's suggestion): H0 for edge j->i = "column j has no extremal dependence on the rest".
Null: circularly shift column j in time by a random offset (keeps its marginal and autocorrelation, breaks cross-dependence),
keep the causal ORDER fixed (estimated once on the real data), re-estimate A, record A[i,j].  p = (1+#null>=obs)/(P+1).
All tested edges are then corrected with Benjamini-Hochberg (FDR) and Bonferroni.
Usage (from python/):
  python perm_edges.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --perms 200
Outputs: ../results/tables/edges_permutation.csv (parent, child, a_hat, p, q_BH, p_bonf, sig_BH)
"""
import argparse, os
import numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1)

def perm_test(X, a=1.3, eps=0.1, delta=0.1, kof=0.11, kaf=0.04, perms=200, seed=0, delta_cand=None):
    """returns (DataFrame of tested edges, A, groups). Candidate edges = thresholded graph (delta_cand or delta)."""
    rng = np.random.default_rng(seed); n = len(X); Xf = to_frechet(X)
    groups = algorithm1(Xf, a, eps, int(kof * n)); kA = int(kaf * n)
    A = algorithm2(Xf, groups, kA)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    cand = sorted(hard_threshold_dag(A, delta if delta_cand is None else delta_cand))
    cols = sorted({j for j, i in cand}); null = {j: [] for j in cols}
    for j in cols:
        for _ in range(perms):
            Xp = Xf.copy(); Xp[:, j] = np.roll(Xf[:, j], int(rng.integers(50, n - 50)))
            Ap = algorithm2(Xp, groups, kA); Ap = Ap / np.maximum(np.linalg.norm(Ap, axis=1, keepdims=True), 1e-12)
            null[j].append(Ap[:, j])
    rows = []
    for j, i in cand:
        nv = np.array([v[i] for v in null[j]]); rows.append((j, i, A[i, j], (1 + (nv >= A[i, j]).sum()) / (perms + 1)))
    df = pd.DataFrame(rows, columns=["parent", "child", "a_hat", "p"])
    if len(df): df["q_BH"] = bh(df.p); df["p_bonf"] = np.minimum(df.p * len(df), 1)
    return df, A, groups

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--perms", type=int, default=200); p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1); p.add_argument("--out", default="../results/tables"); a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names = list(df.columns)
    res, _, _ = perm_test(df.values.astype(float), delta=a.delta, perms=a.perms)
    res["parent"] = [names[j] for j in res.parent]; res["child"] = [names[i] for i in res.child]
    res["sig_BH"] = res.q_BH < a.alpha; res.to_csv(f"{a.out}/edges_permutation.csv", index=False)
    print(f"tested {len(res)} edges; raw p<0.05: {(res.p<0.05).sum()}; BH q<{a.alpha}: {res.sig_BH.sum()}; Bonferroni<0.05: {(res.p_bonf<0.05).sum()}")
    print(res.sort_values("p").head(15).to_string(index=False))
if __name__ == "__main__": main()
