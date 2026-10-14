"""experiments_all.py - simulation experiments for the TA-meeting report (items 3,5,6,7,8,10,12).
Run:  python experiments_all.py        (needs numpy, matplotlib)  -> results.json + fig_*.png
"""
import json, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *

OUT = {}
rng0 = np.random.default_rng(2026)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C1, C2, C3 = "#1f5fa8", "#c2410c", "#4b5563"

# ------------------------------------------------------------------ E1: why a^2-1 and how to choose a
def diff_theory(A, i, j, a):
    return float(np.sum(np.maximum(A[i] ** 2, a * a * A[j] ** 2) - np.maximum(A[i] ** 2, A[j] ** 2)))

def E1():
    d, n, k, R = 8, 2285, 250, 40
    A_true = A_from_C(random_C(d, 0.4, np.random.default_rng(11)))
    anc = lambda i, j: A_true[j, i] > 1e-12 and i != j            # i is an ancestor of j
    sources = [j for j in range(d) if (A_true[j] > 1e-12).sum() == 1]
    nonsrc = [j for j in range(d) if j not in sources]
    pairs_ns = [(i, j) for j in nonsrc for i in range(d) if i != j and anc(i, j)]
    pairs_src = [(i, j) for j in sources for i in range(d) if i != j]
    a_grid = [1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0]
    # simulation of the estimator noise
    est = {a: {p: [] for p in pairs_ns + pairs_src} for a in a_grid}
    for r in range(R):
        X = sample_from_A(A_true, n, np.random.default_rng(100 + r))
        idx, Rr = top_idx(X, k)
        for a in a_grid:
            for (i, j) in pairs_ns + pairs_src:
                s1 = scaling(np.maximum(X[:, i], a * X[:, j]), idx, Rr, d)
                s0 = scaling(np.maximum(X[:, i], X[:, j]), idx, Rr, d)
                est[a][(i, j)].append(s1 - s0)
    rows = []
    for a in a_grid:
        gap = [(a * a - 1) - diff_theory(A_true, i, j, a) for (i, j) in pairs_ns]
        sd_ns = [np.std(est[a][p]) for p in pairs_ns]
        sd_src = [np.std(est[a][p]) for p in pairs_src]
        mean_src = np.mean([np.mean(est[a][p]) for p in pairs_src])
        rows.append(dict(a=a, a2m1=a * a - 1, mean_est_source=float(mean_src),
                         median_gap=float(np.median(gap)), median_sd=float(np.median(sd_ns + sd_src)),
                         snr=float(np.median(gap) / np.median(sd_ns + sd_src))))
    OUT["E1"] = dict(rows=rows, n_sources=len(sources), n_pairs_nonsource=len(pairs_ns), d=d, n=n, k=k, reps=R)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].plot(a_grid, [r["a2m1"] for r in rows], "o--", color=C3, label=r"$a^2-1$ (theory, source $j$)")
    ax[0].plot(a_grid, [r["mean_est_source"] for r in rows], "s-", color=C1, label="estimated, source $j$ (mean)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("a"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("source nodes: diff = $a^2-1$")
    ax[1].plot(a_grid, [r["snr"] for r in rows], "o-", color=C2)
    ax[1].set_xscale("log"); ax[1].set_xlabel("a"); ax[1].set_ylabel("signal / noise")
    ax[1].set_title("separating source from non-source")
    fig.tight_layout(); fig.savefig("fig_a_choice.png"); plt.close(fig)

# ------------------------------------------------------------------ E2: why max <-> extremes
def E2():
    N = 3_000_000
    rng = np.random.default_rng(5)
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    dists = {"Pareto(2) heavy": lambda: rng.pareto(2.0, (N, 2)) + 1.0,
             "Exponential light": lambda: rng.exponential(1.0, (N, 2))}
    for name, gen in dists.items():
        Z = gen(); s = Z.sum(1); m = Z.max(1)
        qs = [0.9, 0.99, 0.999, 0.9999]
        xs = np.quantile(m, qs)
        ratio = [float((s > x).mean() / (m > x).mean()) for x in xs]
        res[name] = dict(q=qs, ratio=ratio)
        ax[0].plot(qs, ratio, "o-", label=name)
    ax[0].set_xscale("logit"); ax[0].set_xlabel("quantile level of max"); ax[0].set_ylabel("P(sum>x) / P(max>x)")
    ax[0].axhline(1, color=C3, lw=0.7, ls=":"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("sum vs. max in the tail")
    # scatter of a small max-linear model: extremes sit on rays
    A = np.array([[0.8, 0.6], [0.0, 1.0]]); X = sample_from_A(A, 4000, np.random.default_rng(1))
    ax[1].loglog(X[:, 0], X[:, 1], ".", ms=2, color=C1, alpha=.5)
    ax[1].set_xlabel("$X_1$"); ax[1].set_ylabel("$X_2$"); ax[1].set_title("max-linear sample, log-log")
    fig.tight_layout(); fig.savefig("fig_max_tail.png"); plt.close(fig)
    OUT["E2"] = res

# ------------------------------------------------------------------ E3/E8: single run vs bootstrap vs rolling window
def block_bootstrap(X, block, rng):
    n = X.shape[0]; nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, nb)
    return np.concatenate([X[s:s + block] for s in starts])[:n]

def truth_edges(A):
    return hard_threshold_dag(A, 0.0)

def E8():
    d, n, R, B, delta = 10, 2285, 24, 40, 0.05
    summary = {}
    stats = {m: [] for m in ["single", "single d=0.10", "single d=0.15", "boot>=0.5", "boot>=0.7", "boot>=0.9"]}
    for r in range(R):
        rng = np.random.default_rng(300 + r)
        A = A_from_C(random_C(d, 0.25, rng)); Tr = truth_edges(A)
        X = sample_from_A(A, n, rng)
        e0, A0, _ = pipeline(X, delta=delta)
        cnt = {}
        for b in range(B):
            Xb = block_bootstrap(X, 20, rng)
            eb, _, _ = pipeline(Xb, delta=delta)
            for e in eb: cnt[e] = cnt.get(e, 0) + 1
        sets = {"single": e0, "single d=0.10": hard_threshold_dag(A0, 0.10), "single d=0.15": hard_threshold_dag(A0, 0.15)}
        for t in (0.5, 0.7, 0.9):
            sets[f"boot>={t}"] = {e for e, c in cnt.items() if c / B >= t}
        for m, es in sets.items():
            p, rc = prec_rec(es, Tr)
            stats[m].append((p, rc, shd(es, Tr), len(es), len(Tr)))
    for m, v in stats.items():
        v = np.array(v)
        summary[m] = dict(precision=float(v[:, 0].mean()), recall=float(v[:, 1].mean()), shd=float(v[:, 2].mean()),
                          n_edges=float(v[:, 3].mean()), n_true=float(v[:, 4].mean()),
                          sd_precision=float(v[:, 0].std()), sd_recall=float(v[:, 1].std()))
    OUT["E8_bootstrap"] = dict(summary=summary, d=d, n=n, reps=R, B=B, delta=delta, block=20)

    # rolling window on data with a structural change in the middle
    d, nh = 8, 2200
    rng = np.random.default_rng(77)
    Cc1 = random_C(d, 0.3, rng); Cc2 = Cc1.copy()
    for i in (1, 2, 3):                                  # rewire parents of three nodes in the second half
        Cc2[i, i + 1:] = 0
        for kk in range(i + 1, d):
            if rng.random() < 0.6: Cc2[i, kk] = rng.uniform(0.4, 1.4)
    A1, A2 = A_from_C(Cc1), A_from_C(Cc2)
    G1, G2 = truth_edges(A1), truth_edges(A2)
    X = np.vstack([sample_from_A(A1, nh, rng), sample_from_A(A2, nh, rng)])
    stat = np.vstack([sample_from_A(A1, 2 * nh, rng)])   # stationary control (no change)
    W, S = 1100, 275
    rows = []
    for name, data in (("change", X), ("stationary", stat)):
        starts = list(range(0, data.shape[0] - W + 1, S)); prev = None
        for s0 in starts:
            e, _, _ = pipeline(data[s0:s0 + W], delta=delta)
            mid = s0 + W / 2
            rows.append(dict(series=name, start=s0, mid=mid, edges=len(e), nshd_G1=nshd(e, G1), nshd_G2=nshd(e, G2),
                             nshd_prev=(nshd(e, prev) if prev is not None else None)))
            prev = e
    OUT["E8_rolling"] = dict(rows=rows, W=W, S=S, n_edges_G1=len(G1), n_edges_G2=len(G2), shd_G1_G2=shd(G1, G2), nshd_G1_G2=nshd(G1, G2))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ms = list(summary.keys())
    x = np.arange(len(ms)); w = 0.38
    ax[0].bar(x - w / 2, [summary[m]["precision"] for m in ms], w, color=C1, label="precision")
    ax[0].bar(x + w / 2, [summary[m]["recall"] for m in ms], w, color=C2, label="recall")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["single\nδ.05", "single\nδ.10", "single\nδ.15", "boot\n≥.5", "boot\n≥.7", "boot\n≥.9"], fontsize=6.5)
    ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("edge recovery vs. truth")
    for name, c in (("change", C1), ("stationary", C3)):
        rr = [r for r in rows if r["series"] == name]
        ax[1].plot([r["mid"] for r in rr], [r["nshd_G1"] for r in rr], "o-", color=c, label=f"{name}: nSHD to G1")
    rr = [r for r in rows if r["series"] == "change"]
    ax[1].plot([r["mid"] for r in rr], [r["nshd_G2"] for r in rr], "s--", color=C2, label="change: nSHD to G2")
    ax[1].axvline(nh, color=C3, lw=0.7, ls=":"); ax[1].legend(frameon=False, fontsize=6.5)
    ax[1].set_xlabel("window midpoint (obs.)"); ax[1].set_title("rolling window")
    fig.tight_layout(); fig.savefig("fig_methods.png"); plt.close(fig)

# ------------------------------------------------------------------ E4: hidden confounder
def E4():
    res = {}
    for label, c2 in (("asymmetric loadings (1.0 vs 0.6)", 0.6), ("symmetric loadings (1.0 vs 1.0)", 1.0)):
        sp = {"1->2": 0, "2->1": 0, "none": 0}; vals = []; R = 150
        for r in range(R):
            rng = np.random.default_rng(900 + r)
            d = 7                                            # node 6 = hidden source, parent of 1 and 2
            C = np.zeros((d, d))
            for i in range(d): C[i, i] = rng.uniform(0.8, 1.2)
            for i in range(6):
                for kk in range(i + 1, 6):
                    if rng.random() < 0.25 and not (i in (1, 2) and kk in (1, 2)): C[i, kk] = rng.uniform(0.4, 1.2)
            C[1, 2] = 0; C[2, 1] = 0
            for i in (1, 2): C[i, 3:6] = 0; C[i, 4:6] = 0       # nodes 1,2 have no observed parents
            C[1, 6] = 1.0; C[2, 6] = c2; C[1, 1] = 1.0; C[2, 2] = 1.0
            A = A_from_C(C); X = sample_from_A(A, 2285, rng)[:, :6]
            # edge 1->2 means "1 is an ancestor of 2" i.e. A[2,1] > 0
            Ahat = algorithm2(to_frechet(X), algorithm1(to_frechet(X), 1.3, 0.15, 250), 92)
            E = hard_threshold_dag(Ahat, 0.05)
            e12, e21 = (1, 2) in E, (2, 1) in E
            sp["1->2" if e12 else ("2->1" if e21 else "none")] += 1 if (e12 != e21) else 0
            if not (e12 or e21): sp["none"] += 0
            vals.append((e12, e21))
        v = np.array(vals)
        res[label] = dict(R=R, spurious_1_to_2=float(v[:, 0].mean()), spurious_2_to_1=float(v[:, 1].mean()),
                          any_spurious=float((v[:, 0] | v[:, 1]).mean()))
    OUT["E4_latent"] = res

# ------------------------------------------------------------------ E5: goodness-of-fit of the max-linear form
def chi_matrix(X, q=0.95):
    n, d = X.shape
    U = np.empty_like(X, dtype=float)
    for i in range(d): U[:, i] = (X[:, i].argsort().argsort() + 1) / (n + 1)
    H = (U > q).astype(float)
    return (H.T @ H) / n / (1 - q)

def gof_stat(X, q=0.95, k_order=None, k_A=None, nsim_mult=5, seed=0):
    Xf = to_frechet(X); n, d = Xf.shape
    g = algorithm1(Xf, 1.3, 0.15, k_order or int(0.1 * n))
    A = algorithm2(Xf, g, k_A or int(0.04 * n))
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Xs = sample_from_A(A, nsim_mult * n, np.random.default_rng(seed))
    c1, c2 = chi_matrix(Xf, q), chi_matrix(Xs, q)
    iu = np.triu_indices(d, 1)
    return float(np.mean(np.abs(c1[iu] - c2[iu])))

def E5():
    d, n, R = 6, 5000, 20
    res = {}
    gens = {}
    def g_rmlm(r):
        rng = np.random.default_rng(1000 + r); return sample_from_A(A_from_C(random_C(d, 0.5, rng)), n, rng)
    def g_sum(r):
        rng = np.random.default_rng(2000 + r); A = A_from_C(random_C(d, 0.5, rng))
        Z = (-np.log(rng.random((n, d)))) ** -0.5; return Z @ A.T                 # SUM instead of max
    def g_t(r):
        rng = np.random.default_rng(3000 + r)
        rho = rng.uniform(0.3, 0.8); S = np.full((d, d), rho); np.fill_diagonal(S, 1.0)
        L = np.linalg.cholesky(S); nu = 3
        return (rng.standard_normal((n, d)) @ L.T) / np.sqrt(rng.chisquare(nu, (n, 1)) / nu)
    for name, g in (("true max-linear (RMLM)", g_rmlm), ("sum-linear heavy-tailed (linear SEM)", g_sum),
                    ("multivariate t (continuous angular measure)", g_t)):
        v = []
        for r in range(R):
            X = g(r)
            v.append(gof_stat(X, seed=r))
        res[name] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), values=[float(x) for x in v])
    OUT["E5_gof"] = dict(res=res, d=d, n=n, reps=R, q=0.95)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    names = list(res.keys())
    ax.boxplot([res[m]["values"] for m in names], tick_labels=["RMLM\n(max)", "linear SEM\n(sum)", "t dist.\n(continuous)"], showfliers=False)
    ax.set_ylabel("mean |chi_data - chi_fit|"); ax.set_title("max-linear goodness-of-fit diagnostic")
    fig.tight_layout(); fig.savefig("fig_gof.png"); plt.close(fig)

if __name__ == "__main__":
    which = sys.argv[1:] or ["E1", "E2", "E8", "E4", "E5"]
    for w in which:
        t = time.time(); globals()[w](); print(w, "done in %.1fs" % (time.time() - t), flush=True); json.dump(OUT, open("results.json", "w"), indent=1)
    json.dump(OUT, open("results.json", "w"), indent=1)
