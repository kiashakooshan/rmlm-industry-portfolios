"""
Faithful Python port of the authors' original R code (CJS_functions_code.R,
from https://github.com/mariokrali). Arrays here use the SAME orientation
as the R code: shape = (d, n)  i.e.  rows = dimensions/nodes, columns = observations,
to minimise transcription risk. See python/rmlm/pipeline.py for an n-by-d wrapper.
"""
import numpy as np


def margins_to_frechet(x):
    """Port of margins_to_frechet(x). x: (d, n). Formula (29) of the paper."""
    d, n = x.shape
    m = np.zeros((d, n))
    for i in range(d):
        xi = x[i, :]
        ranks = xi.argsort().argsort() + 1          # 1..n
        ecdf_vals = ranks / n                         # R's ecdf(x)(x)
        val = (n / (n + 1)) * ecdf_vals
        m[i, :] = (-np.log(val)) ** (-1.0 / 2.0)
    return m


def omega_A(x):
    """x: (m, n_sel) -- already the k selected columns. Returns squared angular
    components (m+1, n_sel), including the dummy zero row (R's rbind(x,0) trick)."""
    xz = np.vstack([x, np.zeros((1, x.shape[1]))])
    denom = (xz ** 2).sum(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return (xz ** 2) / denom


def _topk_cols(x, k):
    c_sum = (x ** 2).sum(axis=0)
    thresh = np.sort(c_sum)[::-1][k - 1]
    return np.where(c_sum >= thresh)[0]


def sigma_M_I(x_full, I, k):
    """Port of sigma_M_I(x, I, k). x_full: (d, n). I: list of 0-indexed rows to
    INCLUDE (R's negative 'exclude' indices are resolved to an include-list by caller)."""
    xi = x_full[I, :]
    idx = _topk_cols(xi, k)
    y = omega_A(xi[:, idx])
    m = len(I)                        # local dimension (mass) -- NOT the global d
    return m * np.mean(y.max(axis=0))


def _complement(d, exclude_set):
    return [i for i in range(d) if i not in exclude_set]


def A_generations(x, k, generations):
    """Port of A_generations(x, k, generations).
    x: (d, n) data, ALREADY PERMUTED so generation 1 (sources, found first by
       causord_eps_a) occupies the LAST rows, generation 2 the next block, etc.
    generations: list of lists of 0-indexed positions in the PERMUTED x.
    Returns A^2 (d, d) -- take sqrt() for A.
    """
    d = x.shape[0]
    b = np.zeros((d, d))
    steps = len(generations)
    aggr = [len(g) for g in generations]
    gen_blocks = []
    for g in range(steps):
        cum_prev = sum(aggr[:g])
        cum_g = sum(aggr[:g + 1])
        gen_blocks.append(list(range(d - cum_g, d - cum_prev)))

    for i in range(d - 1, 0, -1):
        hi = sigma_M_I(x, list(range(i, d)), k) if i <= d - 2 else sigma_M_I(x, [d - 1], k)
        lo = 0.0
        if i <= d - 2:
            lo = sigma_M_I(x, list(range(i + 1, d)), k) if i <= d - 3 else sigma_M_I(x, [d - 1], k)
        b[i, i] = hi - lo
    b[0, 0] = sigma_M_I(x, list(range(d)), k) - np.trace(b)

    for i in range(0, d - 1):
        running = 0.0
        for j in range(i + 1, d):
            if i == 0:
                excl1 = set(range(i + 1, j + 1))
            else:
                excl1 = set(range(0, i)) | set(range(i + 1, j + 1))
            s1 = sigma_M_I(x, _complement(d, excl1), k)
            s2 = sigma_M_I(x, _complement(d, set(range(0 if i > 0 else i, j + 1))), k) if j < d - 1 else 0.0
            b[i, j] = s1 - running - s2
            running += b[i, j]
        diag_save = np.diag(b).copy()
        for blk in gen_blocks:
            b[np.ix_(blk, blk)] = 0.0
        np.fill_diagonal(b, diag_save)

    return b


def mwp_gr_eps(P_orig, eps):
    """Port of mwp_gr_eps: hard-thresholding (section 7.1). P_orig: actual A matrix
    (NOT squared). Returns thresholded A with weak/redundant edges set to 0."""
    d = P_orig.shape[0]
    P = P_orig.copy()
    for m in range(d):
        for i in range(d):
            if i == m:
                continue
            midpath = np.zeros(d)
            for kk in range(d):
                if kk == i or kk == m:
                    continue
                if P_orig[i, kk] > 0 and P_orig[kk, m] > 0 and P_orig[kk, kk] > 0:
                    midpath[kk] = P_orig[i, kk] * P_orig[kk, m] / P_orig[kk, kk]
            if (midpath.max() + eps) >= P_orig[i, m]:
                P[i, m] = 0.0
            else:
                P[i, m] = P_orig[i, m]
    return P


def omega_caus_order(x, k):
    y = (x ** 2).sum(axis=0)
    so = np.sort(y)[::-1][k - 1]
    ind = np.where(y >= so)[0]
    w = (np.abs(x) ** 2 / y[np.newaxis, :])[:, ind]
    return w


def sigma_M(x, k):
    y = omega_caus_order(x, k)
    return 2 * y.max(axis=0).sum() / k


def pair_sigma_diff(z, a, k):
    """z: (2, n) -- rows [node_i, node_j]."""
    v = sigma_M(z, k)
    scaled = (np.array([a, 1.0])[:, None]) * z
    q = (1 + a ** 2) / 2 * sigma_M(scaled, k) - v - (a ** 2 - 1)
    return q


def sigma_i_aj_aI_diff(x, ijI, k, a):
    """x: (d, n). ijI: 0-indexed list [i, j, *I]."""
    i_row = x[[ijI[0]], :]
    jI_rows = a * x[ijI[1:], :]
    y_full = np.vstack([i_row, jI_rows])
    dim_jI = len(ijI) - 1
    dim_i = 1
    y = omega_caus_order(y_full, k)                   # (dim_i+dim_jI, k)
    y3 = np.vstack([y[[0], :], y[1:, :] / (a ** 2)])
    mass = dim_i + (a ** 2) * dim_jI
    p = mass * y.max(axis=0).sum() / k
    ycm = y3[1:, :].max(axis=0)
    pdif = p - mass * np.maximum(y3[0, :], ycm).sum() / k - (a ** 2 - 1) * mass * ycm.sum() / k
    return pdif


def causord_eps_a(x, a, k, eps):
    """Port of causord_eps_a: estimates the causal order. x: (d, n).
    Returns dict{'I': final order (0-indexed, sinks-first..sources-last, matching R
    convention), 'generations': list of 0-indexed groups, generation[0] = sources}."""
    d = x.shape[0]
    Delta = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            Delta[j, i] = pair_sigma_diff(x[[i, j], :], a, k)
        Delta[i, i] = np.inf

    Delta_col_min = Delta.min(axis=0)
    mx = Delta_col_min.max()
    id_ = list(np.where(np.abs(Delta_col_min - mx) <= abs(eps * mx))[0])
    id_ = sorted(id_, key=lambda t: Delta_col_min[t])
    generations = [list(id_)]
    g = 1
    id_set = set(id_)

    while len(id_set) <= d - 2:
        id1 = list(id_set)
        Delta = np.zeros((d, d))
        remaining = [t for t in range(d) if t not in id_set]
        for i in remaining:
            for j in remaining:
                if i == j:
                    continue
                Delta[i, j] = sigma_i_aj_aI_diff(x, [i, j] + list(id_set), k, a)
            Delta[i, i] = np.inf
        for t in id_set:
            Delta[t, :] = np.inf
            Delta[:, t] = np.inf

        Delta_col_min2 = Delta.min(axis=0)
        sub_idx = remaining
        Delta_col_min3 = Delta[np.ix_(sub_idx, sub_idx)].min(axis=0)
        mx3 = Delta_col_min3.max()

        new_id = list(np.where(np.abs(Delta_col_min2 - mx3) <= eps * abs(mx3))[0])
        new_id = sorted(new_id, key=lambda t: Delta_col_min2[t])
        generations.append([t for t in new_id if t not in id1])
        id_set = set(new_id) | set(id1)
        g += 1

    last = [t for t in range(d) if t not in id_set]
    generations.append(last)
    final_order = last + list(id_set)   # R: id <- c(complement, id)

    return {"I": final_order, "generations": generations}
