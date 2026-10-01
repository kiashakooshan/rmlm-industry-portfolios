"""Hard-thresholding + nSHD centroid selection (paper section 7.1)."""
import numpy as np


def hard_threshold_dag(A, delta):
    d = A.shape[0]
    edges = []
    for i in range(d):
        for j in range(d):
            if i == j or A[i, j] <= 0:
                continue
            best_indirect = 0.0
            for kk in range(d):
                if kk != i and kk != j and A[i, kk] > 0 and A[kk, j] > 0 and A[kk, kk] > 0:
                    best_indirect = max(best_indirect, A[i, kk] * A[kk, j] / A[kk, kk])
            if A[i, j] > best_indirect + delta:
                edges.append((j, i))
    return edges


def shd(edges1, edges2):
    s1, s2 = set(edges1), set(edges2)
    return len(s1 - s2) + len(s2 - s1)


def nshd(edges1, edges2):
    denom = len(edges1) + len(edges2)
    return shd(edges1, edges2) / denom if denom > 0 else 0.0


def centroid_graph(graphs):
    scores = []
    for i, g in enumerate(graphs):
        total = sum(nshd(g, graphs[j]) for j in range(len(graphs)) if j != i)
        scores.append(total)
    best = int(np.argmin(scores))
    return graphs[best], best, scores
