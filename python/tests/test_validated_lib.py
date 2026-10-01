"""
Unit + integration tests for rmlm.validated_lib (our primary, from-scratch
implementation). Run with:  pytest python/tests/test_validated_lib.py -v
"""
import numpy as np
import pytest
from rmlm.validated_lib import simulate_rmlm, scaling, algorithm1, algorithm2


@pytest.fixture(scope="module")
def synthetic_data():
    X, A_true, C_true = simulate_rmlm(d=6, n=20000, edge_density=0.35, alpha=2.0, seed=1)
    return X, A_true, C_true


def test_single_node_scaling_is_one(synthetic_data):
    """sigma_i^2 must be close to 1 for every standardised node (Lemma 1(i))."""
    X, A_true, _ = synthetic_data
    R = np.linalg.norm(X, axis=1)
    for i in range(X.shape[1]):
        s = scaling(X[:, i], R, k=500, dim_const=X.shape[1])
        assert 0.8 < s < 1.25, f"node {i}: sigma^2={s} too far from 1"


def test_algorithm1_returns_valid_topological_order(synthetic_data):
    """Every parent (larger original index, by our simulator's convention)
    must appear in an EARLIER or EQUAL group than its children."""
    X, A_true, C_true = synthetic_data
    d = X.shape[1]
    groups = algorithm1(X, a=1.3, eps=0.15, k=500)
    position = {}
    for gi, g in enumerate(groups):
        for node in g:
            position[node] = gi
    for i in range(d):
        parents = [k for k in range(i + 1, d) if C_true[i, k] > 0]
        for p in parents:
            assert position[p] <= position[i], (
                f"parent {p} found later than child {i}: invalid order")


def test_algorithm2_recovers_matrix_reasonably(synthetic_data):
    X, A_true, _ = synthetic_data
    groups = algorithm1(X, a=1.3, eps=0.15, k=500)
    A_est, _ = algorithm2(X, groups, k=500)
    mask = A_true > 0.01
    corr = np.corrcoef(A_true[mask], A_est[mask])[0, 1]
    assert corr > 0.9, f"correlation with ground truth too low: {corr}"
