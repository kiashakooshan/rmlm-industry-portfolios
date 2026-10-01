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


def run(k_values=(90, 92, 94, 96, 98), delta=0.1, a=1.3, eps=0.1):
    df = load_raw()
    X, names, dates = to_losses(df)
    Xf = to_frechet(X)

    graphs = []
    for k in k_values:
        groups = algorithm1(Xf, a=a, eps=eps, k=k)
        A_est, _ = algorithm2(Xf, groups, k=k)
        graphs.append(hard_threshold_dag(A_est, delta))

    best_graph, best_idx, scores = centroid_graph(graphs)
    print(f"chosen k = {k_values[best_idx]}, nSHD scores = {scores}")
    print(f"edges ({len(best_graph)}):")
    for (j, i) in best_graph:
        print(f"  {names[j]} -> {names[i]}")
    return best_graph, names


if __name__ == "__main__":
    run()
