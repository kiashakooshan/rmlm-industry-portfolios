"""dilution_real.py - on YOUR real data: why does adding ONE extra column wreck the graph?
Compares, with N repetitions each (mean over repetitions):
  (a) shuffled DGS10 column, radius from all 31 columns (paper)        -> what you already saw
  (b) shuffled DGS10 column, radius from the original 30 columns only   -> tests the dilution hypothesis
  (c) fresh iid Frechet(2) column (no ties, no volatility clustering), radius from all 31
  (d) same as (c) but radius from the original 30 columns only
Statistics: edges removed from the 30-column graph, whether the extra column is the only source, whether Mines&Coal stay sources.
Usage (from python/):
  python dilution_real.py --csv ../data/processed/X_losses.csv --proxy ../data/raw/dgs10_proxy.csv --start 1989-06-01 --end 1998-06-15 --n 30
"""
import argparse, numpy as np, pandas as pd
import rmlm.fast_lib as L
from rmlm.fast_lib import *
p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--proxy", required=True)
p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--n", type=int, default=30); a = p.parse_args()
df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names, X = list(df.columns), df.values.astype(float); d = len(names)
px = pd.read_csv(a.proxy, index_col=0, parse_dates=True).iloc[:, 0].reindex(df.index).fillna(0).clip(lower=0).values
run = lambda M: pipeline(M, 1.3, 0.1, int(.11 * len(M)), int(.04 * len(M)), 0.1)
L.RADIUS_COLS = None; e0, _, g0 = run(X); E0 = set(e0); rng = np.random.default_rng(0); nn = len(X)
def stat(v, rad):
    L.RADIUS_COLS = rad; e, _, g = run(np.column_stack([X, v])); L.RADIUS_COLS = None
    E = {x for x in e if x[0] < d and x[1] < d}
    return dict(removed=len(E0 - E), extra_only_source=int(g[0] == [d]), mines_coal_sources=int({names.index("Mines"), names.index("Coal")} <= set(g[0])), extra_gen0=int(d in g[0]))
frech = lambda: (-np.log(rng.random(nn))) ** -0.5
rows = {}
for name, gen, rad in [("a shuffled proxy, radius all 31", lambda: rng.permutation(px), None), ("b shuffled proxy, radius orig 30", lambda: rng.permutation(px), d),
                       ("c iid Frechet, radius all 31", frech, None), ("d iid Frechet, radius orig 30", frech, d)]:
    rows[name] = pd.DataFrame([stat(gen(), rad) for _ in range(a.n)]).mean()
print(pd.DataFrame(rows).T.round(2).to_string())
print("\nreading: if (b) and (d) keep Mines&Coal as sources and remove far fewer edges than (a),(c) -> dilution hypothesis supported.")
print("if (c) behaves like (a) -> not about ties/volatility of the proxy. if (a) differs from (c) -> properties of the DGS10 series matter.")
