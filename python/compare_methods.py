"""compare_methods.py - single run vs. block-bootstrap vs. rolling window on YOUR real data.
Usage (from the python/ folder of the repo):
    python compare_methods.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 \
        --window-years 3 --step-years 1 --boot 100
Input csv: daily LOSSES (X = max(-return, 0)), first column = date, one column per industry (output of rmlm.data_prep).
Outputs (in ../results/tables): edges_single.csv, edges_bootstrap.csv, rolling_windows.csv
"""
import argparse, os
import numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--window-years", type=float, default=3.0); p.add_argument("--step-years", type=float, default=1.0)
    p.add_argument("--boot", type=int, default=100); p.add_argument("--block", type=int, default=20)
    p.add_argument("--a", type=float, default=1.3); p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1); p.add_argument("--out", default="../results/tables")
    p.add_argument("--k-order-frac", type=float, default=0.11); p.add_argument("--k-A-frac", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]
    names, X = list(df.columns), df.values.astype(float)
    print(f"data: n={X.shape[0]}, d={X.shape[1]}, {df.index.min().date()} .. {df.index.max().date()}")
    run = lambda M: pipeline(M, a.a, a.eps, int(a.k_order_frac * len(M)), int(a.k_A_frac * len(M)), a.delta)
    fmt = lambda E: sorted((names[j], names[i]) for j, i in E)

    # 1) single run on the whole period
    e0, A0, groups = run(X)
    print("causal order (generations, sources first):")
    for g, grp in enumerate(groups): print("  gen", g, [names[v] for v in grp])
    pd.DataFrame(fmt(e0), columns=["parent", "child"]).to_csv(f"{a.out}/edges_single.csv", index=False)
    print(f"single run: {len(e0)} edges")

    # 2) block bootstrap edge frequencies
    rng = np.random.default_rng(a.seed); cnt = {}
    for b in range(a.boot):
        nb = int(np.ceil(len(X) / a.block)); st = rng.integers(0, len(X) - a.block + 1, nb)
        Xb = np.concatenate([X[s:s + a.block] for s in st])[:len(X)]
        for e in run(Xb)[0]: cnt[e] = cnt.get(e, 0) + 1
    tab = pd.DataFrame([(names[j], names[i], c / a.boot) for (j, i), c in cnt.items()], columns=["parent", "child", "freq"])
    tab = tab.sort_values("freq", ascending=False); tab.to_csv(f"{a.out}/edges_bootstrap.csv", index=False)
    for t in (0.5, 0.7, 0.9): print(f"bootstrap freq >= {t}: {(tab.freq >= t).sum()} edges")

    # 3) rolling windows (margins are re-standardised inside every window)
    W, S = int(a.window_years * 252), int(a.step_years * 252); rows, prev = [], None
    for s0 in range(0, len(X) - W + 1, S):
        e, _, _ = run(X[s0:s0 + W])
        rows.append(dict(start=df.index[s0].date(), end=df.index[s0 + W - 1].date(), n_edges=len(e),
                         nshd_to_previous=(nshd(e, prev) if prev is not None else np.nan),
                         nshd_to_full_sample=nshd(e, e0)))
        prev = e
    pd.DataFrame(rows).to_csv(f"{a.out}/rolling_windows.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
