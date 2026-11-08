"""edge_test.py - CONDITIONAL edge test (parametric bootstrap on the 'edge-absent' boundary model).
For every candidate edge j->i of the estimated graph:
  statistic   S_ij = a_ij - max_k a_ik a_kj / a_kk          (the quantity thresholded with delta in the paper)
  H0          j->i is NOT in the minimum DAG: a_ij equals the best indirect path (S_ij = 0). Dependence between i and j is
              allowed to exist, but only through other paths.
  null model  A0 = A_hat with a_ij replaced by its best indirect path value (rows renormalised). Simulate n days from A0
              (Frechet(2) innovations), keep the causal ORDER fixed, re-estimate A, record S_ij. p = (1+#null>=obs)/(P+1).
All tested edges: Benjamini-Hochberg FDR.
Caveat: A_hat and the candidate list come from the same data (post-selection), so p-values are somewhat optimistic.
Usage (from python/):  python edge_test.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --perms 100
"""
import argparse, os
import numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

def nrm(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
def best_indirect(A, i, j):
    d = A.shape[0]; b = 0.0
    for k in range(d):
        if k != i and k != j and A[i, k] > 0 and A[k, j] > 0 and A[k, k] > 0: b = max(b, A[i, k] * A[k, j] / A[k, k])
    return b
def S_matrix(A):
    d = A.shape[0]; S = np.zeros_like(A)
    for i in range(d):
        for j in range(d):
            if i != j and A[i, j] > 0: S[i, j] = A[i, j] - best_indirect(A, i, j)
    return S
def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; return np.minimum(q, 1)

def edge_test(X, a=1.3, eps=0.1, delta=0.1, kof=0.11, kaf=0.04, perms=100, seed=0):
    rng = np.random.default_rng(seed); n = len(X); Xf = to_frechet(X)
    groups = algorithm1(Xf, a, eps, int(kof * n)); kA = int(kaf * n)
    A = nrm(algorithm2(Xf, groups, kA)); S = S_matrix(A)
    cand = sorted(hard_threshold_dag(A, delta)); rows = []
    for j, i in cand:
        A0 = A.copy(); A0[i, j] = best_indirect(A, i, j); A0 = nrm(A0); nv = np.empty(perms)
        for b in range(perms):
            Xs = to_frechet(sample_from_A(A0, n, rng)); Ab = nrm(algorithm2(Xs, groups, kA)); nv[b] = Ab[i, j] - best_indirect(Ab, i, j)
        rows.append((j, i, A[i, j], S[i, j], (1 + (nv >= S[i, j]).sum()) / (perms + 1)))
    df = pd.DataFrame(rows, columns=["parent", "child", "a_hat", "S_ij", "p"])
    if len(df): df["q_BH"] = bh(df.p)
    return df, A, groups

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--perms", type=int, default=100); p.add_argument("--delta", type=float, default=0.1); p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", default="../results/tables"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names = list(df.columns)
    res, _, _ = edge_test(df.values.astype(float), delta=a.delta, perms=a.perms)
    res["parent"] = [names[j] for j in res.parent]; res["child"] = [names[i] for i in res.child]; res["sig_BH"] = res.q_BH < a.alpha
    res.to_csv(f"{a.out}/edges_conditional_test.csv", index=False)
    print(f"tested {len(res)} edges; raw p<0.05: {(res.p<0.05).sum()}; BH q<{a.alpha}: {res.sig_BH.sum()}")
    print(res.sort_values("p").head(15).to_string(index=False))
if __name__ == "__main__": main()
