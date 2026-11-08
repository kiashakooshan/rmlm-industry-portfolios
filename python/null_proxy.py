"""null_proxy.py - compares the REAL proxy with N shuffled proxies (null), on the same statistics.
Usage (from python/):
  python null_proxy.py --csv ../data/processed/X_losses.csv --proxy ../data/raw/dgs10_proxy.csv --start 1989-06-01 --end 1998-06-15 --n 30
Statistics per run: #edges removed from the no-proxy graph, #edges with PROXY as endpoint, size of generation 0, whether Mines/Coal stay sources.
"""
import argparse, numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *
p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--proxy", required=True)
p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--n", type=int, default=30)
p.add_argument("--a", type=float, default=1.3); p.add_argument("--eps", type=float, default=0.1); p.add_argument("--delta", type=float, default=0.1)
p.add_argument("--out", default="../results/tables"); a = p.parse_args()
df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names, X = list(df.columns), df.values.astype(float)
px = pd.read_csv(a.proxy, index_col=0, parse_dates=True).iloc[:, 0].reindex(df.index).fillna(0).clip(lower=0).values
run = lambda M: pipeline(M, a.a, a.eps, int(.11 * len(M)), int(.04 * len(M)), a.delta)
e0, _, g0 = run(X); E0 = {(names[j], names[i]) for j, i in e0}; d = len(names); nm2 = names + ["PROXY"]
def stats(v):
    e, _, g = run(np.column_stack([X, v])); E = {(nm2[j], nm2[i]) for j, i in e}
    return dict(removed=len(E0 - {x for x in E if "PROXY" not in x}), n_proxy_edges=sum("PROXY" in x for x in E),
                proxy_out=sum(x[0] == "PROXY" for x in E), gen0_size=len(g[0]), proxy_in_gen0="PROXY" in [nm2[v_] for v_ in g[0]],
                mines_coal_still_sources=bool({"Mines", "Coal"} <= {nm2[v_] for v_ in g[0]}))
real = stats(px); rng = np.random.default_rng(0)
null = pd.DataFrame([stats(rng.permutation(px)) for _ in range(a.n)])
print("REAL proxy:", real); print("\nNULL (shuffled proxy) summary:"); print(null.describe().loc[["mean", "std", "min", "max"]].to_string())
for k in ["removed", "n_proxy_edges", "proxy_out"]:
    print(f"{k}: real={real[k]}  null mean={null[k].mean():.1f}  share of null >= real = {(null[k] >= real[k]).mean():.2f}")
print("proxy_in_gen0 in null:", null.proxy_in_gen0.mean(), "| mines&coal still sources in null:", null.mines_coal_still_sources.mean())
pd.concat([pd.DataFrame([real]), null]).to_csv(f"{a.out}/null_proxy.csv", index=False)
