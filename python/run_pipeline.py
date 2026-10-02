"""
End-to-end pipeline runner: raw data -> Frechet margins -> causal order ->
A matrix -> hard-thresholded DAG -> centroid DAG across a few k values -> plot.
Run as:  python run_pipeline.py
"""
import numpy as np
from rmlm.data_prep import load_raw, to_losses, to_frechet
from rmlm.validated_lib import algorithm1, algorithm2
from rmlm.thresholding import hard_threshold_dag, centroid_graph

NAMES_30 = ["Food","Beer","Smoke","Games","Books","Hshld","Clths","Hlth","Chems","Txtls",
            "Cnstr","Steel","FabPr","ElcEq","Autos","Carry","Mines","Coal","Oil","Util",
            "Telcm","Servs","BusEq","Paper","Trans","Whlsl","Rtail","Meals","Fin","Other"]


# Published causal order from the paper (section 7.1), sink-first .. source-last.
# ("ElcEq"/"FabPr" corrected from the PDF-extraction artefacts "Elcq"/"Fabr".)
PAPER_ORDER = ["Whlsl","Hshld","Food","Txtls","Smoke","BusEq","Hlth","Carry","Beer","Servs",
               "Util","Cnstr","Rtail","Clths","Telcm","Paper","Chems","ElcEq","Meals","Other",
               "Games","Fin","FabPr","Steel","Trans","Books","Autos","Oil","Mines","Coal"]


def kendall_tau(rank_a, rank_b):
    """Simple O(n^2) Kendall's tau, no scipy dependency. rank_* : dict name -> rank."""
    names = list(rank_a.keys())
    n = len(names)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[names[i]] - rank_a[names[j]]
            db = rank_b[names[i]] - rank_b[names[j]]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def run(k_order=250, k_values=(90, 92, 94, 96, 98), delta=0.1, a=1.3, eps=0.1):
    df = load_raw()
    X, names, dates = to_losses(df)
    Xf = to_frechet(X)

    # Step 1 (paper section 7.1): estimate the causal order ONCE, with the
    # larger k_order=250 -- do NOT re-estimate the order for every k below.
    groups = algorithm1(Xf, a=a, eps=eps, k=k_order)
    order_position = {}
    for gi, g in enumerate(groups):
        for idx in g:
            order_position[names[idx]] = gi
    # compare against the paper's published order (sink-first .. source-last)
    paper_rank = {name: pos for pos, name in enumerate(PAPER_ORDER)}
    our_rank = {name: order_position.get(name, -1) for name in PAPER_ORDER}
    tau = kendall_tau(paper_rank, our_rank)
    print(f"estimated generations ({len(groups)} groups):")
    for gi, g in enumerate(groups):
        print(f"  gen {gi}: {[names[i] for i in g]}")
    print(f"\nKendall's tau vs. the paper's published order: {tau:.3f}  (1.0 = perfect match)")

    # Step 2: with the order FIXED, estimate the A matrix (and threshold it)
    # for each k in k_values, then pick the centroid DAG -- this is the part
    # of the paper that is actually meant to vary with k.
    graphs = []
    for k in k_values:
        A_est, _ = algorithm2(Xf, groups, k=k)
        graphs.append(hard_threshold_dag(A_est, delta))

    best_graph, best_idx, scores = centroid_graph(graphs)
    print(f"\nchosen k = {k_values[best_idx]}, nSHD scores = {[round(s,3) for s in scores]}")
    print(f"edges ({len(best_graph)}):")
    for (j, i) in best_graph:
        print(f"  {names[j]} -> {names[i]}")
    return best_graph, names, groups, tau


if __name__ == "__main__":
    run()
