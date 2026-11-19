"""coef_fixed_order.py - are the coefficient intervals wide because the ORDER changes? Same block-bootstrap resamples, two analyses:
FULL  = whole pipeline re-run in every resample (order re-estimated)   FIXED = order of the original data kept, only A and the threshold re-estimated.
For the most frequent edges: bootstrap interval of a_ij under both, selection frequency, and the 'meaningful' rule
(freq >= 0.7 and 2.5% quantile of a_ij > 0.05).
Usage (from python/):  python coef_fixed_order.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100 --topk 15
Outputs -> ../results/coef_fixed_order/ : coef_full_vs_fixed.csv, fig_coef_full_vs_fixed.png, fig_interval_width.png, summary.txt
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

def nrm(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--boot", type=int, default=100); p.add_argument("--block", type=int, default=20); p.add_argument("--topk", type=int, default=15)
    p.add_argument("--delta", type=float, default=0.1); p.add_argument("--out", default="../results/coef_fixed_order"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end]; names = list(df.columns); X = df.values.astype(float); n, d = X.shape
    ko, kA = int(.11 * n), int(.04 * n); rng = np.random.default_rng(a.seed)
    g0 = algorithm1(to_frechet(X), 1.3, 0.1, ko); e0 = hard_threshold_dag(nrm(algorithm2(to_frechet(X), g0, kA)), a.delta)
    AF, AX = np.zeros((a.boot, d, d)), np.zeros((a.boot, d, d)); EF, EX = np.zeros((a.boot, d, d), bool), np.zeros((a.boot, d, d), bool)
    for b in range(a.boot):
        st = rng.integers(0, n - a.block + 1, int(np.ceil(n / a.block))); Xb = np.concatenate([X[s:s + a.block] for s in st])[:n]
        e, A, _ = pipeline(Xb, 1.3, 0.1, ko, kA, a.delta); AF[b] = A
        for j, i in e: EF[b, i, j] = True
        A2 = nrm(algorithm2(to_frechet(Xb), g0, kA)); AX[b] = A2
        for j, i in hard_threshold_dag(A2, a.delta): EX[b, i, j] = True
    fF, fX = EF.mean(0), EX.mean(0); rows = []
    for i in range(d):
        for j in range(d):
            if i != j and (fF[i, j] > 0 or fX[i, j] > 0):
                xf, xx = AF[:, i, j], AX[:, i, j]
                rows.append(dict(parent=names[j], child=names[i], _i=i, _j=j, freq_full=fF[i, j], freq_fixed=fX[i, j],
                                 full_mean=xf.mean(), full_q025=np.quantile(xf, .025), full_q975=np.quantile(xf, .975), full_width=np.quantile(xf, .975) - np.quantile(xf, .025),
                                 fixed_mean=xx.mean(), fixed_q025=np.quantile(xx, .025), fixed_q975=np.quantile(xx, .975), fixed_width=np.quantile(xx, .975) - np.quantile(xx, .025)))
    T = pd.DataFrame(rows); T["meaningful_full"] = (T.freq_full >= .7) & (T.full_q025 > .05); T["meaningful_fixed"] = (T.freq_fixed >= .7) & (T.fixed_q025 > .05)
    T = T.sort_values("freq_fixed", ascending=False); T.drop(columns=["_i", "_j"]).to_csv(f"{a.out}/coef_full_vs_fixed.csv", index=False)
    out = open(f"{a.out}/summary.txt", "w")
    def say(s): print(s); out.write(s + "\n")
    say(f"B={a.boot}, block={a.block}; edges considered: {len(T)}")
    say(f"edges with freq>=0.7: FULL {(T.freq_full>=.7).sum()}  FIXED {(T.freq_fixed>=.7).sum()}")
    say(f"'meaningful' edges (freq>=0.7 and 2.5% quantile > 0.05): FULL {T.meaningful_full.sum()}  FIXED {T.meaningful_fixed.sum()}")
    top = T.head(a.topk); say(f"top-{a.topk} edges by fixed-order frequency: mean interval width FULL {top.full_width.mean():.3f}  FIXED {top.fixed_width.mean():.3f}; mean 2.5% quantile FULL {top.full_q025.mean():.3f}  FIXED {top.fixed_q025.mean():.3f}")
    say(top[["parent", "child", "freq_full", "freq_fixed", "full_q025", "full_q975", "fixed_q025", "fixed_q975"]].round(2).to_string(index=False)); out.close()
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(top) + 1.6)); H = 0.36
    for r, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        for off, c, lab, xs in ((H / 2, "#b45309", "full (order re-estimated)", AF[:, row._i, row._j]), (-H / 2, "#14325c", "fixed order", AX[:, row._i, row._j])):
            bp = ax.boxplot(xs, positions=[r + off], vert=False, widths=H * .9, showfliers=False, patch_artist=True, whis=(2.5, 97.5))
            for bx in bp["boxes"]: bx.set(facecolor=c, alpha=.4, edgecolor=c)
            for k in ("whiskers", "caps", "medians"):
                for ln in bp[k]: ln.set_color(c)
        ax.text(1.01, r, f"freq {row.freq_full:.2f} / {row.freq_fixed:.2f}", va="center", fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{x} → {y}" for x, y in zip(top.parent[::-1], top.child[::-1])], fontsize=8); ax.set_xlabel("coefficient a_ij over bootstrap resamples (whiskers = 95% interval)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#b45309", alpha=.4), plt.Rectangle((0, 0), 1, 1, fc="#14325c", alpha=.4)], labels=["full (order re-estimated)", "fixed order"], fontsize=8, loc="lower right")
    ax.set_title("Coefficient spread: order re-estimated vs order held fixed (right: selection frequency full / fixed)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_coef_full_vs_fixed.png", dpi=200); plt.close()
    fig, ax = plt.subplots(figsize=(4.6, 4.4)); ax.scatter(top.full_width, top.fixed_width, c="#14325c"); m = max(top.full_width.max(), top.fixed_width.max()) * 1.05
    ax.plot([0, m], [0, m], "k--", lw=.8); ax.set_xlabel("95% interval width, full"); ax.set_ylabel("95% interval width, fixed order"); ax.set_title("each point = one edge (below line: narrower with fixed order)", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_interval_width.png", dpi=200); plt.close(); print("saved to", a.out)

if __name__ == "__main__":
    main()
