"""event_study.py - professor's idea: pick an extreme day and see what happens to the graph.
For each event date (default: the 3 days with the largest radius R in the window) it reports
  (a) INFLUENCE: nSHD(graph on all data, graph with +-h days around the event removed)  [h=10]
      compared with the same quantity for h days removed at RANDOM places (null, 50 draws)
  (b) BEFORE/AFTER: nSHD(graph before event, graph after event) compared with random split points (null)
Usage (from python/):
  python event_study.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15
  python event_study.py ... --dates 1997-10-27 1989-10-13
"""
import argparse, os
import numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *
p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
p.add_argument("--dates", nargs="*"); p.add_argument("--h", type=int, default=10); p.add_argument("--null", type=int, default=50)
p.add_argument("--out", default="../results/tables"); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; X = df.values.astype(float); n = len(X)
run = lambda M: pipeline(M, 1.3, 0.1, int(.11 * len(M)), int(.04 * len(M)), 0.1)[0]
E0 = run(X); rng = np.random.default_rng(0)
R = np.linalg.norm(to_frechet(X), axis=1)
ev = [df.index.get_loc(pd.Timestamp(d)) for d in a.dates] if a.dates else list(np.argsort(-R)[:3])
def drop(t): return np.delete(X, np.arange(max(t - a.h, 0), min(t + a.h + 1, n)), axis=0)
null_infl = [nshd(E0, run(drop(int(rng.integers(a.h, n - a.h))))) for _ in range(a.null)]
rows = []
for t in ev:
    infl = nshd(E0, run(drop(t)))
    sp = [nshd(run(X[:s]), run(X[s:])) for s in rng.integers(int(.25 * n), int(.75 * n), a.null)]
    ba = nshd(run(X[:t]), run(X[t:])) if .2 * n < t < .8 * n else np.nan
    rows.append(dict(date=df.index[t].date(), radius=round(float(R[t]), 2), nshd_remove_event=round(infl, 3),
                     null_remove_mean=round(float(np.mean(null_infl)), 3), null_remove_95=round(float(np.quantile(null_infl, .95)), 3),
                     p_remove=round((1 + sum(v >= infl for v in null_infl)) / (a.null + 1), 3),
                     nshd_before_after=round(ba, 3) if ba == ba else None, null_split_mean=round(float(np.mean(sp)), 3)))
out = pd.DataFrame(rows); out.to_csv(f"{a.out}/event_study.csv", index=False); print(out.to_string(index=False))
print("reading: p_remove small => removing this event changes the graph more than removing 21 random days (the event is influential).")
