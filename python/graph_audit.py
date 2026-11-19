"""graph_audit.py - TA tips: (1) draw the graph in every rolling window; (2) which graph, taken as THE graph, is closest to all the others
(smallest expected distance)?; (3) are the relations directed or undirected (skeleton vs direction stability)?
Usage (from python/):
  python graph_audit.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100
Outputs -> ../results/graph_audit/ : fig_window_graphs.png, fig_candidate_distance.png, fig_skeleton_vs_directed.png,
  candidate_distances.csv, pair_freq.csv, window_pairwise_nshd.csv, summary.txt
Definitions: nSHD(E1,E2)=|E1 xor E2|/(|E1|+|E2|).  'expected error' of a candidate graph G = mean nSHD(G, B_b) over the bootstrap graphs B_b
(a descriptive criterion: the closest graph to the cloud of graphs that resampling could have produced).
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

def boot_sample(X, block, rng):
    n = len(X); st = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([X[s:s + block] for s in st])[:n]

def skel(E): return {frozenset(e) for e in E}

def snshd(E1, E2):
    a, b = skel(E1), skel(E2); t = len(a) + len(b)
    return len(a ^ b) / t if t else 0.0

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--boot", type=int, default=100); p.add_argument("--block", type=int, default=20)
    p.add_argument("--window-years", type=float, default=3.0); p.add_argument("--step-years", type=float, default=1.0)
    p.add_argument("--a", type=float, default=1.3); p.add_argument("--eps", type=float, default=0.1); p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--out", default="../results/graph_audit"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names = list(df.columns); X = df.values.astype(float); n, d = X.shape
    ko, kA = int(.11 * n), int(.04 * n)
    run = lambda M: pipeline(M, a.a, a.eps, int(.11 * len(M)), int(.04 * len(M)), a.delta)
    out = open(f"{a.out}/summary.txt", "w")
    def say(s): print(s); out.write(s + "\n")

    # ---- graphs: single run, paper-style centroid over k, bootstrap, rolling windows
    e_single, A0, g0 = run(X)
    Xf = to_frechet(X); groups = algorithm1(Xf, a.a, a.eps, ko)
    kgraphs = {}
    for k in (90, 92, 94, 96, 98):
        A = algorithm2(Xf, groups, k); A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12); kgraphs[k] = hard_threshold_dag(A, a.delta)
    ks = list(kgraphs); D = np.array([[nshd(kgraphs[i], kgraphs[j]) for j in ks] for i in ks]); k_best = ks[int(D.sum(1).argmin())]
    e_paper = kgraphs[k_best]                                  # paper's rule: centroid (min total nSHD) among k-variants, order estimated once
    BG, BF = [], {}
    for b in range(a.boot):
        e, _, _ = run(boot_sample(X, a.block, rng)); BG.append(e)
        for x in e: BF[x] = BF.get(x, 0) + 1
    cons5 = [x for x, v in BF.items() if v / a.boot >= .5]; cons7 = [x for x, v in BF.items() if v / a.boot >= .7]
    Dm = np.array([[nshd(BG[i], BG[j]) for j in range(a.boot)] for i in range(a.boot)]); medoid = BG[int(Dm.sum(1).argmin())]
    W, S = int(a.window_years * 252), int(a.step_years * 252); WG, wl, wgroups = [], [], []
    for s0 in range(0, n - W + 1, S):
        e, _, g = run(X[s0:s0 + W]); WG.append(e); wgroups.append(g); wl.append(f"{df.index[s0].year}-{df.index[s0+W-1].year}")

    # ---- (2) which graph is closest to all the others?
    cands = {"single run (k=91)": e_single, f"paper-style centroid (k={k_best})": e_paper, "bootstrap medoid": medoid,
             "consensus freq>=0.5": cons5, "consensus freq>=0.7": cons7}
    for l, e in zip(wl, WG): cands[f"window {l}"] = e
    rows = []
    for nm, e in cands.items():
        db = [nshd(e, g) for g in BG]; dw = [nshd(e, g) for g in WG if g is not e]
        rows.append(dict(candidate=nm, n_edges=len(e), mean_nshd_to_bootstrap=np.mean(db), sd=np.std(db), mean_nshd_to_windows=np.mean(dw), mean_skeleton_nshd_to_bootstrap=np.mean([snshd(e, g) for g in BG])))
    C = pd.DataFrame(rows).sort_values("mean_nshd_to_bootstrap"); C.to_csv(f"{a.out}/candidate_distances.csv", index=False)
    say("== Which graph is closest to the others? (mean nSHD to the %d bootstrap graphs; lower = closer) ==" % a.boot); say(C.round(3).to_string(index=False))
    say(f"\nbootstrap graphs among themselves: mean nSHD = {Dm[np.triu_indices(a.boot,1)].mean():.3f}")
    Wm = np.array([[nshd(x, y) for y in WG] for x in WG]); pd.DataFrame(Wm, index=wl, columns=wl).round(3).to_csv(f"{a.out}/window_pairwise_nshd.csv")
    wmed = wl[int(Wm.sum(1).argmin())]; say(f"window graph closest to the other windows: {wmed}; mean pairwise nSHD between windows = {Wm[np.triu_indices(len(WG),1)].mean():.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); Cc = C.iloc[::-1]
    ax[0].barh(Cc.candidate, Cc.mean_nshd_to_bootstrap, xerr=Cc.sd, color=["#b45309" if "paper" in c or "single" in c else "#14325c" for c in Cc.candidate]); ax[0].set_xlabel("mean nSHD to the bootstrap graphs (lower = closer to all)"); ax[0].tick_params(axis="y", labelsize=7)
    im = ax[1].imshow(Wm, cmap="viridis"); ax[1].set_xticks(range(len(wl))); ax[1].set_yticks(range(len(wl))); ax[1].set_xticklabels(wl, rotation=60, fontsize=7); ax[1].set_yticklabels(wl, fontsize=7); plt.colorbar(im, ax=ax[1]); ax[1].set_title("nSHD between rolling-window graphs", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_candidate_distance.png", dpi=200); plt.close()

    # ---- (3) directed or undirected?
    pr = []
    for i in range(d):
        for j in range(i + 1, d):
            f1 = BF.get((i, j), 0) / a.boot; f2 = BF.get((j, i), 0) / a.boot
            fs = np.mean([((i, j) in set(g)) or ((j, i) in set(g)) for g in BG]) if (f1 + f2) > 0 else 0.0
            if fs > 0: pr.append(dict(node_a=names[i], node_b=names[j], freq_a_to_b=f1, freq_b_to_a=f2, freq_skeleton=fs, direction_consistency=max(f1, f2) / fs))
    P = pd.DataFrame(pr).sort_values("freq_skeleton", ascending=False); P.to_csv(f"{a.out}/pair_freq.csv", index=False)
    cnt = lambda s, t: int((s >= t).sum()); dirf = np.concatenate([P.freq_a_to_b.values, P.freq_b_to_a.values])
    say("\n== Directed vs undirected (bootstrap) ==")
    for t in (.5, .7, .9): say(f"  threshold {t}: directed edges {cnt(dirf,t)}   undirected adjacencies {cnt(P.freq_skeleton.values,t)}")
    g = P[P.freq_skeleton >= .5]; say(f"  pairs adjacent in >=50% of resamples: {len(g)}; mean direction consistency (share of the majority direction) = {g.direction_consistency.mean():.2f}; pairs with consistency < 0.75: {(g.direction_consistency<.75).sum()}")
    sk_prev = [snshd(WG[i], WG[i - 1]) for i in range(1, len(WG))]; di_prev = [nshd(WG[i], WG[i - 1]) for i in range(1, len(WG))]
    say(f"  rolling windows, consecutive-window distance: directed nSHD mean {np.mean(di_prev):.2f} ({min(di_prev):.2f}-{max(di_prev):.2f}); skeleton nSHD mean {np.mean(sk_prev):.2f} ({min(sk_prev):.2f}-{max(sk_prev):.2f})")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); ts = np.linspace(.3, .95, 14)
    ax[0].plot(ts, [cnt(dirf, t) for t in ts], "o-", label="directed edges"); ax[0].plot(ts, [cnt(P.freq_skeleton.values, t) for t in ts], "s--", label="undirected adjacencies"); ax[0].set_xlabel("frequency threshold"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=8)
    x = np.arange(1, len(WG)); ax[1].plot(x, di_prev, "o-", label="directed nSHD"); ax[1].plot(x, sk_prev, "s--", label="skeleton nSHD"); ax[1].set_xticks(x); ax[1].set_xticklabels(wl[1:], rotation=60, fontsize=7); ax[1].legend(fontsize=8); ax[1].set_title("distance to previous window", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_skeleton_vs_directed.png", dpi=200); plt.close()

    # ---- (1) draw graph in every window (layers = generations, 0 = source on top)
    keyn = {"Mines", "Coal", "BusEq", "Servs", "Oil", "Whlsl", "Util"}
    fig, axs = plt.subplots(2, 4, figsize=(20, 9)); axs = axs.ravel()
    def draw(ax, E, groups, title):
        pos = {}
        for gi, nodes in enumerate(groups):
            for k, v in enumerate(sorted(nodes, key=lambda v: names[v])): pos[v] = ((k + .5) / len(nodes), -gi)
        for (j, i) in E: ax.annotate("", xy=pos[i], xytext=pos[j], arrowprops=dict(arrowstyle="-", color="#14325c", alpha=.18, lw=.6))
        for v, (x_, y_) in pos.items():
            hot = names[v] in keyn; ax.plot(x_, y_, "o", color="#b45309" if hot else "#14325c", ms=5 if hot else 3)
            if hot: ax.text(x_, y_ + .12, names[v], fontsize=7, ha="center", color="#b45309")
        ax.set_title(title, fontsize=9); ax.axis("off")
    draw(axs[0], e_paper, g0, f"full sample 1989-1998 ({len(e_paper)} edges)")
    for k in range(len(WG)): draw(axs[k + 1], WG[k], wgroups[k], f"{wl[k]} ({len(WG[k])} edges)")
    for k in range(len(WG) + 1, 8): axs[k].axis("off")
    fig.suptitle("Graph in each rolling window: rows = generations (top = sources), lines = edges (direction: higher row to lower row); highlighted: Mines, Coal, BusEq, Servs, Oil, Whlsl, Util", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_window_graphs.png", dpi=170); plt.close(); out.close(); print("saved to", a.out)

if __name__ == "__main__":
    main()
