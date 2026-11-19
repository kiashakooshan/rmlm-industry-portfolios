"""lag_check.py - TA tip: with what LAG are two nodes related? (the RMLM is instantaneous; this checks whether lead-lag structure exists)
For an edge parent j -> child i and lag h in -H..H:  chi_h = P( U_i(t+h) > q | U_j(t) > q ),  U = rank/(n+1), q = 0.95.
h > 0 means the parent leads. Chance level is 1-q = 0.05. If the model were purely instantaneous, chi_h for h != 0 would be near chance and
chi_h would not be larger for h>0 than for h<0 (lead-lag asymmetry would be ~0).
Usage: python lag_check.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --edges ../results/tables/edges_bootstrap.csv --top 15
Outputs -> ../results/lag_check/ : lag_table.csv, lag_by_window.csv, fig_lag.png, summary.txt
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def chi_lag(Ui, Uj, h, q):
    n = len(Ui); ex_j = Uj > q
    if h >= 0: a, b = ex_j[: n - h], Ui[h:] > q
    else: a, b = ex_j[-h:], Ui[: n + h] > q
    return float((a & b).sum() / max(a.sum(), 1))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--edges", required=True); p.add_argument("--top", type=int, default=15); p.add_argument("--H", type=int, default=5); p.add_argument("--q", type=float, default=.95)
    p.add_argument("--window-years", type=float, default=3.0); p.add_argument("--step-years", type=float, default=1.0); p.add_argument("--out", default="../results/lag_check")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; n = len(df)
    U = df.apply(lambda c: pd.Series(rankdata(c.values, method="average") / (len(c) + 1), index=c.index)); E = pd.read_csv(a.edges).sort_values("freq", ascending=False).head(a.top)
    lags = list(range(-a.H, a.H + 1)); rows = []
    for _, r in E.iterrows():
        row = dict(parent=r.parent, child=r.child, boot_freq=r.freq)
        for h in lags: row[f"h={h:+d}"] = chi_lag(U[r.child].values, U[r.parent].values, h, a.q)
        row["lead_minus_lag"] = np.mean([row[f"h={h:+d}"] for h in range(1, a.H + 1)]) - np.mean([row[f"h={-h:+d}"] for h in range(1, a.H + 1)])
        rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(f"{a.out}/lag_table.csv", index=False)
    cols = [f"h={h:+d}" for h in lags]; M = T[cols].values
    out = open(f"{a.out}/summary.txt", "w")
    def say(s): print(s); out.write(s + "\n")
    say(f"chance level = {1-a.q:.2f}; {len(T)} most frequent edges; chi_h = P(child extreme at t+h | parent extreme at t)")
    say("mean chi_h over edges: " + "  ".join(f"h={h:+d}:{M[:, k].mean():.3f}" for k, h in enumerate(lags)))
    say(f"peak lag is 0 for {(np.array(lags)[M.argmax(1)]==0).sum()} of {len(T)} edges; mean chi_0 = {T['h=+0'].mean():.3f}; mean chi over |h|=1..{a.H} = {np.mean([M[:,k].mean() for k,h in enumerate(lags) if h!=0]):.3f}")
    say(f"lead-lag asymmetry (mean chi parent-leads minus child-leads, h=1..{a.H}): mean {T.lead_minus_lag.mean():+.4f}, positive for {(T.lead_minus_lag>0).sum()} of {len(T)} edges")
    W, S = int(a.window_years * 252), int(a.step_years * 252); wr = []
    for s0 in range(0, n - W + 1, S):
        Uw = U.iloc[s0:s0 + W].apply(lambda c: rankdata(c.values) / (W + 1)); lab = f"{df.index[s0].year}-{df.index[s0+W-1].year}"
        vals = np.array([[chi_lag(Uw[r.child].values, Uw[r.parent].values, h, a.q) for h in lags] for _, r in E.iterrows()]).mean(0)
        wr.append(dict(window=lab, **{c: v for c, v in zip(cols, vals)}))
    Wd = pd.DataFrame(wr); Wd.to_csv(f"{a.out}/lag_by_window.csv", index=False); say("\nmean chi_h by window (top edges):"); say(Wd.round(3).to_string(index=False)); out.close()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for _, r in T.iterrows(): ax[0].plot(lags, [r[c] for c in cols], alpha=.35, lw=.8)
    ax[0].plot(lags, M.mean(0), "k-o", lw=2); ax[0].axhline(1 - a.q, color="r", ls="--", lw=.8); ax[0].set_xlabel("lag h (>0: parent leads)"); ax[0].set_ylabel("chi_h"); ax[0].set_title("lagged tail dependence, top edges (black = mean, red = chance)", fontsize=9)
    ax[1].bar(range(len(T)), T.lead_minus_lag.values); ax[1].set_xticks(range(len(T))); ax[1].set_xticklabels([f"{a_}→{b_}" for a_, b_ in zip(T.parent, T.child)], rotation=75, fontsize=7); ax[1].set_title("lead-lag asymmetry (parent leads - child leads)", fontsize=9)
    im = ax[2].imshow(Wd[cols].values, aspect="auto", cmap="viridis"); ax[2].set_xticks(range(len(cols))); ax[2].set_xticklabels(lags); ax[2].set_yticks(range(len(Wd))); ax[2].set_yticklabels(Wd.window, fontsize=7); plt.colorbar(im, ax=ax[2]); ax[2].set_title("mean chi_h by window", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_lag.png", dpi=200); plt.close(); print("saved to", a.out)

if __name__ == "__main__":
    main()
