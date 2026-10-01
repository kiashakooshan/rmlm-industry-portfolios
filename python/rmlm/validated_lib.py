import numpy as np

# ---------- simulate a ground-truth RMLM ----------
def simulate_rmlm(d, n, edge_density=0.35, alpha=2.0, seed=0):
    rng = np.random.default_rng(seed)
    C = np.zeros((d, d))
    for i in range(d):
        for k in range(i+1, d):          # k has larger index -> k can be a parent of i
            if rng.random() < edge_density:
                C[i, k] = rng.uniform(0.3, 1.4)
        C[i, i] = rng.uniform(0.5, 1.3)
    A = np.zeros((d, d))
    for i in reversed(range(d)):          # process sources (large index) first
        A[i, i] = C[i, i]
        for k in range(i+1, d):
            if C[i, k] > 0:
                A[i, :] = np.maximum(A[i, :], C[i, k] * A[k, :])
        A[i, i] = max(A[i, i], C[i, i])
    # standardize rows to unit L2 norm (Assumption A3 / Lemma 1(i), valid because alpha=2)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    U = rng.random((n, d))
    Z = (-np.log(U)) ** (-1.0/alpha)      # standard Frechet(alpha)
    X = np.zeros((n, d))
    for i in range(d):
        X[:, i] = np.max(A[i, :][None, :] * Z, axis=1)
    return X, A, C


def scaling(Mvals, R, k, dim_const):
    """empirical estimator (28)/(30) generalised to any max-projection"""
    idx = np.argsort(-R)[:k]
    return dim_const * np.mean((Mvals[idx] / R[idx])**2)   # = (dim_const/k) * sum(...)


def build_M(X, cols, weight=1.0):
    if len(cols) == 0:
        return None
    return weight * X[:, cols].max(axis=1)


def combine(*parts):
    parts = [p for p in parts if p is not None]
    out = parts[0]
    for p in parts[1:]:
        out = np.maximum(out, p)
    return out


# ---------- Algorithm 1: causal order ----------
def algorithm1(X, a=1.3, eps=0.15, k=250):
    n, d = X.shape
    R = np.linalg.norm(X, axis=1)
    remaining = list(range(d))
    order_groups = []          # list of lists, group 0 = sources found first
    O = []                     # already ordered (found) nodes
    while remaining:
        if len(remaining) == 1:
            order_groups.append(list(remaining))
            break
        col_min = {}
        for j in remaining:
            best = np.inf
            for i in remaining:
                if i == j:
                    continue
                Mij_O   = combine(build_M(X, [i]), build_M(X, [j]), build_M(X, O))
                MiajO   = combine(build_M(X, [i]), build_M(X, [j], a), build_M(X, O, a))
                MjO     = combine(build_M(X, [j]), build_M(X, O))
                s1 = scaling(MiajO, R, k, d)
                s2 = scaling(Mij_O, R, k, d)
                s3 = scaling(MjO,  R, k, d)
                delta = s1 - s2 - (a**2 - 1)*s3
                best = min(best, delta)
            col_min[j] = best
        best_val = max(col_min.values())
        thresh = eps * abs(best_val) if best_val != 0 else eps
        chosen = [j for j in remaining if col_min[j] >= best_val - thresh]
        order_groups.append(sorted(chosen))
        O = O + chosen
        remaining = [j for j in remaining if j not in chosen]
    return order_groups   # groups[0] = true sources ... groups[-1] = final sinks


# ---------- Algorithm 2: coefficient matrix ----------
def algorithm2(X, order_groups, k):
    n, d = X.shape
    R = np.linalg.norm(X, axis=1)
    # new index: group found FIRST (sources) -> HIGHEST paper-index
    flat_new_to_old = []
    for g in reversed(order_groups):   # reversed: last group (sinks) first -> lowest index
        flat_new_to_old.extend(g)
    # flat_new_to_old[0] = a sink (paper index 1), flat_new_to_old[-1] = a source (paper index d)
    Xp = X[:, flat_new_to_old]
    dp = Xp.shape[1]
    A2 = np.zeros((dp, dp))
    for i in range(dp-1):
        cols_i_to_end = list(range(i, dp))
        cols_ip1_to_end = list(range(i+1, dp))
        s_i = scaling(build_M(Xp, cols_i_to_end), R, k, d)
        s_ip1 = scaling(build_M(Xp, cols_ip1_to_end), R, k, d) if cols_ip1_to_end else 0.0
        A2[i, i] = max(s_i - s_ip1, 0)
        running = A2[i, i]
        for j in range(i+1, dp-1):
            cols_j1_end = list(range(j+1, dp))
            s_a = scaling(build_M(Xp, [i]+cols_j1_end), R, k, d)
            s_b = scaling(build_M(Xp, cols_j1_end), R, k, d) if cols_j1_end else 0.0
            val = s_a - s_b - running
            A2[i, j] = max(val, 0)
            running += A2[i, j]
        A2[i, dp-1] = max(1.0 - running, 0)
    A2[dp-1, dp-1] = 1.0
    A_est = np.sqrt(A2)
    # re-map back to original column order
    A_full = np.zeros((d, d))
    for newi, oldi in enumerate(flat_new_to_old):
        for newj, oldj in enumerate(flat_new_to_old):
            A_full[oldi, oldj] = A_est[newi, newj]
    return A_full, flat_new_to_old
