"""gof_check.py - is the max-linear form compatible with your data? (tail-dependence diagnostic)
Usage: python gof_check.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --reps 30
Idea: fit the RMLM, simulate from the fitted model, compare pairwise upper-tail dependence chi_ij(q) of the data and of the model.
The reference distribution comes from data simulated from the FITTED model itself (parametric bootstrap)."""
import argparse
import numpy as np, pandas as pd
try:
    from rmlm.fast_lib import *
except ImportError:
    from rmlm_lib import *
import experiments_all as E

p = argparse.ArgumentParser(); p.add_argument("--csv", required=True); p.add_argument("--start"); p.add_argument("--end")
p.add_argument("--reps", type=int, default=30); p.add_argument("--q", type=float, default=0.95)
a = p.parse_args()
X = pd.read_csv(a.csv, index_col=0, parse_dates=True).loc[a.start:a.end].values.astype(float)
n, d = X.shape
stat = E.gof_stat(X, q=a.q)
Xf = to_frechet(X); A = algorithm2(Xf, algorithm1(Xf, 1.3, 0.1, int(0.11 * n)), int(0.04 * n))
A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
ref = [E.gof_stat(sample_from_A(A, n, np.random.default_rng(r)), q=a.q, seed=r) for r in range(a.reps)]
print(f"observed statistic = {stat:.4f}")
print(f"reference (data simulated from the fitted max-linear model): mean={np.mean(ref):.4f}, 95th pct={np.quantile(ref, .95):.4f}")
print("=> observed above the 95th percentile suggests the max-linear form does not fully capture the extremal dependence." if stat > np.quantile(ref, .95) else "=> observed within the range expected under the fitted max-linear model.")
