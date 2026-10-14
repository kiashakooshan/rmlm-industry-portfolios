"""audit_extra.py - two audits for the proposal, on YOUR real data.
 (1) split-half reproducibility: cut the series into blocks, odd blocks vs even blocks, compare the two graphs (nSHD),
     repeated over several random block lengths/offsets; compared with a bootstrap-filtered version (freq>=0.7).
 (2) proxy test: add an external column (csv: date, value >= 0, e.g. |daily change of DGS10| or WTI loss) and see which
     edges disappear / which nodes change generation.
Usage (from python/):
  python audit_extra.py split --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15
  python audit_extra.py proxy --csv ../data/processed/X_losses.csv --proxy ../data/raw/dgs10_abs_change.csv --start 1989-06-01 --end 1998-06-15
"""
import argparse, os
import numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

def mk_run(a, n):  # settings as in the paper
    return lambda M: pipeline(M, a.a, a.eps, int(a.k_order_frac * len(M)), int(a.k_A_frac * len(M)), a.delta)

def boot_filter(run, X, B, block, thr, rng):
    cnt = {}
    for _ in range(B):
        nb = int(np.ceil(len(X) / block)); st = rng.integers(0, len(X) - block + 1, nb)
        Xb = np.concatenate([X[s:s + block] for s in st])[:len(X)]
        for e in run(Xb)[0]: cnt[e] = cnt.get(e, 0) + 1
    return {e for e, c in cnt.items() if c / B >= thr}

def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["split", "proxy"])
    p.add_argument("--csv", required=True); p.add_argument("--proxy"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--a", type=float, default=1.3); p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1); p.add_argument("--k-order-frac", type=float, default=0.11)
    p.add_argument("--k-A-frac", type=float, default=0.04); p.add_argument("--reps", type=int, default=10)
    p.add_argument("--boot", type=int, default=50); p.add_argument("--block", type=int, default=20)
    p.add_argument("--out", default="../results/tables"); p.add_argument("--seed", type=int, default=0); p.add_argument("--shuffle", action="store_true", help="control: randomly permute the proxy in time")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]
    names, X = list(df.columns), df.values.astype(float); run = mk_run(a, len(X))
    if a.mode == "split":
        rows = []
        for r in range(a.reps):
            L = int(rng.integers(40, 120)); off = int(rng.integers(0, L))   # random block length / offset
            idx = (np.arange(len(X)) + off) // L
            h1, h2 = X[idx % 2 == 0], X[idx % 2 == 1]
            e1, e2 = run(h1)[0], run(h2)[0]
            f1 = boot_filter(run, h1, a.boot, a.block, 0.7, rng); f2 = boot_filter(run, h2, a.boot, a.block, 0.7, rng)
            rows.append(dict(block=L, n1=len(e1), n2=len(e2), nshd_raw=nshd(e1, e2),
                             n1_f=len(f1), n2_f=len(f2), nshd_boot07=nshd(f1, f2)))
            print(rows[-1])
        t = pd.DataFrame(rows); t.to_csv(f"{a.out}/split_half.csv", index=False)
        print(t[["nshd_raw", "nshd_boot07"]].describe().loc[["mean", "std"]])
        print("lower nshd = more reproducible. If nshd_boot07 < nshd_raw, bootstrap filtering helps on real data.")
    else:
        px = pd.read_csv(a.proxy, index_col=0, parse_dates=True).iloc[:, 0].reindex(df.index).fillna(0).clip(lower=0)
        if a.shuffle: px = pd.Series(rng.permutation(px.values), index=px.index)
        X2 = np.column_stack([X, px.values]); names2 = names + ["PROXY"]
        e0, _, g0 = run(X); e1, _, g1 = run(X2)
        E0 = {(names[j], names[i]) for j, i in e0}; E1 = {(names2[j], names2[i]) for j, i in e1}
        gone = sorted(E0 - E1); new = sorted(E1 - E0)
        print("edges without proxy:", len(E0), " with proxy:", len(E1), " nSHD(no-proxy, proxy minus proxy edges):",
              nshd({(j, i) for j, i in e1 if j < len(names) and i < len(names)}, e0))
        print("generation 0 without proxy:", [names[v] for v in g0[0]]); print("generation 0 with proxy:", [names2[v] for v in g1[0]])
        print("removed edges:", gone); print("proxy edges:", [e for e in new if "PROXY" in e])
        pd.DataFrame({"removed": [f"{x}->{y}" for x, y in gone]}).to_csv(f"{a.out}/proxy_removed_edges.csv", index=False)
if __name__ == "__main__": main()
