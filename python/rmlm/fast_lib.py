"""rmlm_lib.py - self-contained implementation of the RMLM tools (Klueppelberg & Krali).
Convention (same as the paper): X is an (n, d) array; node with LARGER index may be a parent of a smaller one.
A[i, j] != 0  <=>  j is an ancestor of i (or j == i).  Edge tuples are (parent, child).
"""
import numpy as np


# ---------------------------------------------------------------- simulation
def A_from_C(C):
    """Max-linear coefficient matrix from edge weights C (eq. (2) of the paper), rows normalised to unit norm."""
    d = C.shape[0]
    A = np.zeros((d, d))
    for i in reversed(range(d)):
        A[i, i] = C[i, i]
        for k in range(i + 1, d):
            if C[i, k] > 0:
                A[i, :] = np.maximum(A[i, :], C[i, k] * A[k, :])
        A[i, i] = max(A[i, i], C[i, i])
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def random_C(d, density=0.35, rng=None):
    rng = np.random.default_rng(rng)
    C = np.zeros((d, d))
    for i in range(d):
        C[i, i] = rng.uniform(0.5, 1.3)
        for k in range(i + 1, d):
            if rng.random() < density:
                C[i, k] = rng.uniform(0.3, 1.4)
    return C


def sample_from_A(A, n, rng=None, alpha=2.0):
    rng = np.random.default_rng(rng)
    d = A.shape[0]
    Z = (-np.log(rng.random((n, d)))) ** (-1.0 / alpha)      # standard Frechet(alpha)
    X = np.empty((n, d))
    for i in range(d):
        X[:, i] = (A[i][None, :] * Z).max(axis=1)
    return X


def simulate_rmlm(d, n, edge_density=0.35, alpha=2.0, seed=0):
    rng = np.random.default_rng(seed)
    C = random_C(d, edge_density, rng)
    A = A_from_C(C)
    return sample_from_A(A, n, rng, alpha), A, C


def to_frechet(X, alpha=2.0):
    """eq. (29): empirical-cdf transform of every column to standard Frechet(alpha)."""
    n, d = X.shape
    U = np.empty_like(X, dtype=float)
    for i in range(d):
        U[:, i] = (X[:, i].argsort().argsort() + 1) / (n + 1)
    return (-np.log(U)) ** (-1.0 / alpha)


# ---------------------------------------------------------------- scalings
def top_idx(X, k):
    R = np.linalg.norm(X, axis=1)
    return np.argsort(-R)[:k], R


def scaling(Mvals, idx, R, d):
    """hat sigma^2_M  =  (d/k) * sum over k largest radii of (M/R)^2   (eq. (28)/(30))"""
    return d * np.mean((Mvals[idx] / R[idx]) ** 2)


def _mx(*parts):
    parts = [p for p in parts if p is not None]
    out = parts[0]
    for p in parts[1:]:
        out = np.maximum(out, p)
    return out


# ---------------------------------------------------------------- Algorithm 1
def algorithm1(X, a=1.3, eps=0.15, k=250):
    """Estimate the causal order. Returns list of generations; generation 0 = sources."""
    n, d = X.shape
    idx, R = top_idx(X, k)
    remaining, O, groups = list(range(d)), [], []
    while remaining:
        if len(remaining) == 1:
            groups.append(list(remaining)); break
        MO = X[:, O].max(axis=1) if O else None
        col_min = {}
        for j in remaining:
            Xj = X[:, j]
            Mj_O = _mx(Xj, MO)
            s3 = scaling(Mj_O, idx, R, d)
            best = np.inf
            for i in remaining:
                if i == j:
                    continue
                Xi = X[:, i]
                s2 = scaling(_mx(Xi, Mj_O), idx, R, d)
                s1 = scaling(_mx(Xi, a * Xj, None if MO is None else a * MO), idx, R, d)
                best = min(best, s1 - s2 - (a ** 2 - 1) * s3)
            col_min[j] = best
        m = max(col_min.values())
        thr = eps * abs(m) if m != 0 else eps
        chosen = [j for j in remaining if col_min[j] >= m - thr]
        groups.append(sorted(chosen))
        O += chosen
        remaining = [j for j in remaining if j not in chosen]
    return groups


# ---------------------------------------------------------------- Algorithm 2
def algorithm2(X, groups, k):
    """Estimate the max-linear coefficient matrix A respecting the order in `groups`."""
    n, d = X.shape
    idx, R = top_idx(X, k)
    perm = [v for g in reversed(groups) for v in g]       # sinks first ... sources last
    Xp = X[:, perm]
    A2 = np.zeros((d, d))
    s = lambda cols: scaling(Xp[:, cols].max(axis=1), idx, R, d)
    for i in range(d - 1):
        a_ii = s(list(range(i, d))) - (s(list(range(i + 1, d))) if i + 1 < d else 0.0)
        A2[i, i] = max(a_ii, 0)
        run = A2[i, i]
        for j in range(i + 1, d - 1):
            rest = list(range(j + 1, d))
            val = s([i] + rest) - s(rest) - run
            A2[i, j] = max(val, 0)
            run += A2[i, j]
        A2[i, d - 1] = max(1.0 - run, 0)
    A2[d - 1, d - 1] = 1.0
    Ap = np.sqrt(A2)
    A = np.zeros((d, d))
    for ni, oi in enumerate(perm):
        for nj, oj in enumerate(perm):
            A[oi, oj] = Ap[ni, nj]
    return A


# ---------------------------------------------------------------- graph helpers
def hard_threshold_dag(A, delta):
    """edge j->i kept iff A[i,j] > max_k A[i,k]A[k,j]/A[k,k] + delta  (section 7.1)."""
    d = A.shape[0]
    edges = set()
    for i in range(d):
        for j in range(d):
            if i == j or A[i, j] <= 0:
                continue
            best = 0.0
            for m in range(d):
                if m not in (i, j) and A[i, m] > 0 and A[m, j] > 0 and A[m, m] > 0:
                    best = max(best, A[i, m] * A[m, j] / A[m, m])
            if A[i, j] > best + delta:
                edges.add((j, i))
    return edges


def reach_edges(A, tol=1e-9):
    """reachability graph: every positive off-diagonal entry is an edge (ancestor -> descendant)."""
    d = A.shape[0]
    return {(j, i) for i in range(d) for j in range(d) if i != j and A[i, j] > tol}


def shd(e1, e2):
    return len(set(e1) ^ set(e2))


def nshd(e1, e2):
    t = len(e1) + len(e2)
    return shd(e1, e2) / t if t else 0.0


def prec_rec(est, truth):
    est, truth = set(est), set(truth)
    tp = len(est & truth)
    return (tp / len(est) if est else 1.0), (tp / len(truth) if truth else 1.0)


def pipeline(X, a=1.3, eps=0.15, k_order=None, k_A=None, delta=0.05, standardise=True):
    """full pipeline on an (n,d) array: [Frechet] -> order -> A -> thresholded DAG."""
    Xf = to_frechet(X) if standardise else X
    n = Xf.shape[0]
    k_order = k_order or max(20, int(0.1 * n))
    k_A = k_A or max(15, int(0.04 * n))
    groups = algorithm1(Xf, a, eps, k_order)
    A = algorithm2(Xf, groups, k_A)
    return hard_threshold_dag(A, delta), A, groups
