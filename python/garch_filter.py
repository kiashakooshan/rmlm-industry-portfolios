"""garch_filter.py - professor's idea: remove volatility clustering with a GARCH(1,1) filter, then re-run the audits.
Fits a Gaussian-QMLE GARCH(1,1) (own implementation, no extra packages) to each industry's RETURNS in the window,
takes standardised residuals z_t = e_t / sigma_t and defines filtered losses X_f = max(-z, 0).
Usage (from python/):
  python garch_filter.py --raw ../data/raw/30_Industry_Portfolios_Daily.csv --start 1989-06-01 --end 1998-06-15 --out ../data/processed/X_losses_garch.csv
"""
import argparse, io
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

def read_french(path):
    lines, found = [], False
    for ln in open(path, encoding="utf-8", errors="ignore"):
        s = ln.strip()
        if not found:
            if ",Food" in s.replace(" ", ""): found = True; lines.append(ln)
            continue
        if not s or "Equal Weighted" in ln or "Copyright" in ln: break
        lines.append(ln)
    df = pd.read_csv(io.StringIO("".join(lines)), index_col=0, na_values=[-99.99, -999])
    df.columns = df.columns.str.strip(); df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    return df[~df.index.isna()].apply(pd.to_numeric, errors="coerce")

def sigma2_path(e, w, al, be):
    n = len(e); s0 = np.var(e)
    drive = np.empty(n); drive[0] = 0.0; drive[1:] = w + al * e[:-1] ** 2
    # s_t = be*s_{t-1} + drive_t, with s_0 = s0   (drive[0] unused)
    zi = [be * s0]
    out, _ = lfilter([1.0], [1.0, -be], drive[1:], zi=zi)
    return np.concatenate([[s0], out])

def fit_garch(r):
    e = r - r.mean(); v = np.var(e)
    def nll(p):
        w, al, be = p
        if al + be >= 0.9999: return 1e10
        s2 = sigma2_path(e, w, al, be)
        if np.any(s2 <= 0): return 1e10
        return 0.5 * np.sum(np.log(s2) + e ** 2 / s2)
    best = None
    for al0, be0 in [(0.05, 0.90), (0.10, 0.80), (0.03, 0.95)]:
        res = minimize(nll, [v * (1 - al0 - be0), al0, be0], method="L-BFGS-B", bounds=[(1e-8, None), (0, 0.5), (0, 0.9999)])
        if best is None or res.fun < best.fun: best = res
    w, al, be = best.x; s2 = sigma2_path(e, w, al, be)
    return e / np.sqrt(s2), (w, al, be)

def acf1(x): x = x - x.mean(); return float((x[1:] * x[:-1]).sum() / (x * x).sum())

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--raw", required=True); p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--out", required=True)
    a = p.parse_args()
    R = read_french(a.raw).loc[a.start:a.end]; R = R.fillna(0.0)
    Z = pd.DataFrame(index=R.index, columns=R.columns, dtype=float); pers, a1_before, a1_after = [], [], []
    for c in R.columns:
        z, (w, al, be) = fit_garch(R[c].values.astype(float)); Z[c] = z; pers.append(al + be)
        a1_before.append(acf1(R[c].values ** 2)); a1_after.append(acf1(z ** 2))
    Xf = np.maximum(-Z, 0.0); Xf.to_csv(a.out)
    print(f"n={len(Z)}, d={Z.shape[1]}; mean persistence alpha+beta = {np.mean(pers):.3f} (min {np.min(pers):.2f}, max {np.max(pers):.2f})")
    print(f"lag-1 autocorrelation of squared series: before filter mean={np.mean(a1_before):.3f}, after filter mean={np.mean(a1_after):.3f}")
    print("wrote", a.out)
