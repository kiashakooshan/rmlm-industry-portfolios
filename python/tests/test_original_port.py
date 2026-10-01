"""
Smoke + cross-check tests for rmlm.original_port (faithful port of the
authors' R code). These are intentionally LOOSER than test_validated_lib.py:
our own tests show this port has higher estimation variance on small
synthetic examples (see docs/06_known_discrepancy.md) -- this is flagged
as an open research question (gap #10 in docs/implementation_and_gaps_fa.md),
not swept under the rug. If you tighten these thresholds after your own
debugging, please also update that doc.
"""
import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from rmlm.validated_lib import simulate_rmlm
from rmlm.original_port import causord_eps_a, A_generations, mwp_gr_eps, margins_to_frechet


@pytest.fixture(scope="module")
def synthetic_data_dn():
    X, A_true, C_true = simulate_rmlm(d=6, n=40000, edge_density=0.35, alpha=2.0, seed=1)
    return X.T, A_true, C_true   # (d, n) orientation, as the R code expects


def test_margins_to_frechet_runs_and_is_positive():
    raw = np.random.default_rng(0).standard_normal((5, 3000)) ** 2
    out = margins_to_frechet(raw)
    assert out.shape == raw.shape
    assert np.all(out > 0)
    assert not np.isnan(out).any()


def test_causord_returns_valid_topological_order(synthetic_data_dn):
    X, A_true, C_true = synthetic_data_dn
    d = X.shape[0]
    out = causord_eps_a(X, a=1.3, k=1500, eps=0.2)
    # position of each node in the generations list
    position = {}
    for gi, g in enumerate(out["generations"]):
        for node in g:
            position[node] = gi
    for i in range(d):
        parents = [k for k in range(i + 1, d) if C_true[i, k] > 0]
        for p in parents:
            assert position[p] <= position[i]


def test_A_generations_smoke_and_loose_accuracy(synthetic_data_dn):
    X, A_true, C_true = synthetic_data_dn
    d = X.shape[0]
    k = 1500
    out = causord_eps_a(X, a=1.3, k=k, eps=0.2)
    perm = out["I"]
    X_perm = X[perm, :]
    A2 = A_generations(X_perm, k, out["generations"])
    A_est_perm = np.sqrt(np.maximum(A2, 0))
    A_full = np.zeros((d, d))
    for newi, oldi in enumerate(perm):
        for newj, oldj in enumerate(perm):
            A_full[oldi, oldj] = A_est_perm[newi, newj]
    mask = A_true > 0.01
    corr = np.corrcoef(A_true[mask], A_full[mask])[0, 1]
    # loose threshold on purpose -- see module docstring
    assert corr > 0.6, f"correlation unexpectedly low: {corr}"


def test_mwp_gr_eps_removes_redundant_edges():
    # a tiny hand-built example matching the paper's Example 1 (section 2):
    # a13 == a12*a23 exactly -> the direct edge 3->1 must be removed at eps=0
    A = np.array([
        [0.5, 0.6, 0.3],   # a13 = 0.3 = a12*a23 = 0.6*0.5
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ])
    P = mwp_gr_eps(A, eps=0.0)
    assert P[0, 2] == 0.0   # redundant edge removed (a13 == a12*a23)
    assert P[0, 1] == 0.6   # genuine edge kept
