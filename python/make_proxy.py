"""make_proxy.py - builds a proxy csv (date,value>=0) for audit_extra.py proxy.
Usage (from python/):  python make_proxy.py dgs10   |   python make_proxy.py wti
If the download fails: open https://fred.stlouisfed.org/series/DGS10 (or DCOILWTICO) -> Download -> CSV,
save it as ../data/raw/DGS10.csv (or WTI.csv) and re-run the same command.
"""
import sys, os, pandas as pd, numpy as np
which = sys.argv[1].lower()
sid, local = {"dgs10": ("DGS10", "../data/raw/DGS10.csv"), "wti": ("DCOILWTICO", "../data/raw/WTI.csv")}[which]
if os.path.exists(local): df = pd.read_csv(local)
else: df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
df.columns = ["date", "v"]; df["date"] = pd.to_datetime(df["date"]); df["v"] = pd.to_numeric(df["v"], errors="coerce")
s = df.set_index("date")["v"].dropna()
out = s.diff().abs() if which == "dgs10" else (-s.pct_change()).clip(lower=0)   # |change in yield| ; oil LOSS
out = out.dropna(); out.name = "value"
os.makedirs("../data/raw", exist_ok=True); f = f"../data/raw/{which}_proxy.csv"; out.to_csv(f)
print("wrote", f, len(out), "rows", out.index.min().date(), "..", out.index.max().date())
