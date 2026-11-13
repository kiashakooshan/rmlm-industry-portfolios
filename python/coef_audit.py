"""coef_audit.py - TA request: (1) in every bootstrap resample, how do the COEFFICIENTS a_ij change and are they meaningful?
(2) in every rolling window, what features does the graph show?
Usage (from python/):
  python coef_audit.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100 --win-boot 30 --topk 15
Outputs (../results/coef_audit): figures (png) + csv tables.
Definitions (descriptive, NOT formal tests):
  a_ij      = coefficient of parent j in child i (row of A, unit row norm), estimated in each resample / window (0 if estimated <= 0)
  S_ij      = a_ij - max_k a_ik a_kj / a_kk    (the quantity compared with delta in thresholding)
  freq      = fraction of resamples with edge j->i kept (S_ij > delta)
  meaningful= freq >= 0.7  AND  2.5th percentile of a_ij > 0.05   ("stable & coefficient clearly away from zero")
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

def S_of(A):
    d = A.shape[0]; diag = np.maximum(np.diag(A), 1e-12)
    best = np.zeros((d, d))
    for m in range(d):
        c = A[:, m][:, None] * A[m, :][None, :] / diag[m]
        c[m, :] = 0; c[:, m] = 0              # exclude m == i and m == j
        best = np.maximum(best, c)
    return A - best

def rank_of(groups, d):
    r = np.zeros(d)
    for g, nodes in enumerate(groups):
        for v in nodes: r[v] = g
    return r

def boot_sample(X, block, rng):
    n = len(X); st = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([X[s:s + block] for s in st])[:n]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--boot", type=int, default=100); p.add_argument("--win-boot", type=int, default=30)
    p.add_argument("--block", type=int, default=20); p.add_argument("--topk", type=int, default=15)
    p.add_argument("--window-years", type=float, default=3.0); p.add_argument("--step-years", type=float, default=1.0)
    p.add_argument("--a", type=float, default=1.3); p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1); p.add_argument("--out", default="../results/coef_audit")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]
    names, X = list(df.columns), df.values.astype(float); n, d = X.shape
    run = lambda M: pipeline(M, a.a, a.eps, int(.11 * len(M)), int(.04 * len(M)), a.delta)
    rng = np.random.default_rng(a.seed)
    e0, A0, g0 = run(X); r0 = rank_of(g0, d)
    order = sorted(range(d), key=lambda v: (r0[v], names[v]))            # sources first
    print(f"n={n}, d={d}; full-sample edges={len(e0)}; sources={[names[v] for v in g0[0]]}")

    # ---------- PART 1: every bootstrap resample ----------
    As = np.zeros((a.boot, d, d)); Ss = np.zeros((a.boot, d, d)); Es = np.zeros((a.boot, d, d), bool)
    for b in range(a.boot):
        e, A, g = run(boot_sample(X, a.block, rng))
        As[b] = A; Ss[b] = S_of(A)
        for j, i in e: Es[b, i, j] = True
    freq = Es.mean(0)
    rows = []
    for i in range(d):
        for j in range(d):
            if i == j or (freq[i, j] == 0 and (j, i) not in e0): continue
            x = As[:, i, j]
            rows.append(dict(parent=names[j], child=names[i], in_full_sample_graph=(j, i) in e0, full_sample_a=A0[i, j],
                             boot_freq=freq[i, j], a_mean=x.mean(), a_sd=x.std(), a_q025=np.quantile(x, .025), a_q975=np.quantile(x, .975),
                             S_mean=Ss[:, i, j].mean(), S_q025=np.quantile(Ss[:, i, j], .025), _i=i, _j=j))
    T = pd.DataFrame(rows)
    T["meaningful"] = (T.boot_freq >= 0.7) & (T.a_q025 > 0.05)
    T = T.sort_values(["boot_freq", "a_mean"], ascending=False)
    T.drop(columns=["_i", "_j"]).to_csv(f"{a.out}/boot_edge_coef.csv", index=False)
    print(f"edges with freq>=0.7: {(T.boot_freq>=.7).sum()}; 'meaningful' (freq>=0.7 & CI>0.05): {T.meaningful.sum()}")
    top = T.head(a.topk)

    # fig 1: distribution of a_ij over resamples for top edges
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(top) + 1.5))
    for r, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        x = As[:, row._i, row._j]; col = "#2a7f4f" if row.meaningful else "#b45309"
        bp = ax.boxplot(x, positions=[r], vert=False, widths=.6, showfliers=False, patch_artist=True)
        for bx in bp["boxes"]: bx.set(facecolor=col, alpha=.35, edgecolor=col)
        ax.plot(row.full_sample_a, r, "k|", ms=14)
        ax.text(1.02, r, f"freq={row.boot_freq:.2f}", va="center", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{pp} → {cc}" for pp, cc in zip(top.parent[::-1], top.child[::-1])], fontsize=8)
    ax.set_xlabel("coefficient a_ij across bootstrap resamples (black tick = full-sample value)")
    ax.set_title("Bootstrap coefficient spread per edge (green = freq≥0.7 and 95% interval above 0.05; orange = not)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_boot_coef_box.png", dpi=200); plt.close()

    # fig 2: coefficient in every resample (trace), top 6 edges
    fig, axs = plt.subplots(3, 2, figsize=(10, 7), sharex=True); axs = axs.ravel()
    for k, (_, row) in enumerate(top.head(6).iterrows()):
        x = As[:, row._i, row._j]; axs[k].plot(range(1, a.boot + 1), x, ".-", lw=.6, ms=3)
        axs[k].axhline(row.full_sample_a, color="k", ls="--", lw=.8)
        axs[k].set_title(f"{row.parent} → {row.child} (freq {row.boot_freq:.2f})", fontsize=8)
    for ax in axs[-2:]: ax.set_xlabel("bootstrap resample b")
    fig.suptitle("Coefficient a_ij in each bootstrap resample (dashed = full sample)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_boot_coef_trace.png", dpi=200); plt.close()

    # fig 3: heatmaps mean / sd / freq (nodes ordered by full-sample generation)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, M, t in zip(axs, [As.mean(0), As.std(0), freq], ["mean a_ij", "sd of a_ij", "edge frequency"]):
        Mo = M[np.ix_(order, order)].copy(); np.fill_diagonal(Mo, np.nan)
        im = ax.imshow(Mo, cmap="viridis", vmin=0); ax.set_title(t, fontsize=9); plt.colorbar(im, ax=ax, fraction=.046)
        ax.set_xticks(range(d)); ax.set_yticks(range(d)); ax.set_xticklabels([names[v] for v in order], rotation=90, fontsize=5); ax.set_yticklabels([names[v] for v in order], fontsize=5)
        ax.set_xlabel("parent j"); ax.set_ylabel("child i")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_boot_heatmaps.png", dpi=200); plt.close()

    # ---------- PART 2: every rolling window ----------
    W, Sx = int(a.window_years * 252), int(a.step_years * 252)
    key = [(row._j, row._i) for _, row in top.head(8).iterrows()]
    feat, erows, prev, labels, ranks = [], [], None, [], []
    for s0 in range(0, n - W + 1, Sx):
        Xw = X[s0:s0 + W]; e, A, g = run(Xw); r = rank_of(g, d); lab = f"{df.index[s0].year}-{df.index[s0+W-1].year}"
        Aw = np.zeros((a.win_boot, d, d)); Ew = np.zeros((a.win_boot, d, d), bool)
        for b in range(a.win_boot):
            eb, Ab, _ = run(boot_sample(Xw, a.block, rng)); Aw[b] = Ab
            for j, i in eb: Ew[b, i, j] = True
        strong = sorted(e, key=lambda t: -A[t[1], t[0]])[:5]
        feat.append(dict(window=lab, start=df.index[s0].date(), end=df.index[s0 + W - 1].date(), n_edges=len(e), n_generations=len(g),
                         sources="; ".join(names[v] for v in g[0]), strongest_edges="; ".join(f"{names[j]}→{names[i]} ({A[i,j]:.2f})" for j, i in strong),
                         mean_coef_of_edges=np.mean([A[i, j] for j, i in e]) if e else np.nan,
                         nshd_to_previous=nshd(e, prev) if prev is not None else np.nan, nshd_to_full_sample=nshd(e, e0),
                         kendall_tau_order_vs_full=kendalltau(r, r0)[0],
                         edges_stable_in_window_boot_ge07=int((Ew.mean(0) >= .7).sum())))
        for j, i in key:
            x = Aw[:, i, j]
            erows.append(dict(window=lab, parent=names[j], child=names[i], a_window=A[i, j], boot_q025=np.quantile(x, .025), boot_q975=np.quantile(x, .975),
                              boot_freq=Ew[:, i, j].mean(), edge_in_window_graph=(j, i) in e))
        prev = e; labels.append(lab); ranks.append(r)
    F = pd.DataFrame(feat); F.to_csv(f"{a.out}/rolling_features.csv", index=False); E = pd.DataFrame(erows); E.to_csv(f"{a.out}/rolling_edge_coef.csv", index=False)
    print(F.drop(columns=["start", "end"]).to_string(index=False))

    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5)); axs = axs.ravel()
    for k, (j, i) in enumerate(key):
        s = E[(E.parent == names[j]) & (E.child == names[i])]; x = np.arange(len(s))
        ax = axs[k]; ax.errorbar(x, s.a_window, yerr=[np.maximum(s.a_window - s.boot_q025, 0), np.maximum(s.boot_q975 - s.a_window, 0)], fmt="o-", ms=4, lw=1, capsize=2)
        ax.axhline(A0[i, j], color="k", ls="--", lw=.7); ax.set_xticks(x); ax.set_xticklabels(s.window, rotation=60, fontsize=6)
        ax.set_title(f"{names[j]} → {names[i]}", fontsize=8); ax.set_ylim(bottom=0)
    fig.suptitle("Coefficient of key edges in each rolling window (bars = within-window bootstrap 95% interval, dashed = full sample)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_roll_coef.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(8, 7)); R = np.array(ranks).T[order]
    im = ax.imshow(R, cmap="viridis_r", aspect="auto"); plt.colorbar(im, label="generation (0 = source)")
    ax.set_yticks(range(d)); ax.set_yticklabels([names[v] for v in order], fontsize=6); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_title("Causal-order generation of each industry in each rolling window", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_roll_generation.png", dpi=200); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3)); x = np.arange(len(F))
    axs[0].plot(x, F.n_edges, "o-"); axs[0].set_title("number of edges", fontsize=9)
    axs[1].plot(x, F.nshd_to_previous, "o-", label="to previous window"); axs[1].plot(x, F.nshd_to_full_sample, "s--", label="to full sample"); axs[1].legend(fontsize=7); axs[1].set_title("nSHD", fontsize=9)
    axs[2].plot(x, F.kendall_tau_order_vs_full, "o-"); axs[2].set_title("Kendall τ of order vs full-sample order", fontsize=9)
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels(F.window, rotation=60, fontsize=6)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_roll_summary.png", dpi=200); plt.close()
    print("saved to", a.out)

if __name__ == "__main__":
    main()
