"""
Data preparation: raw 30-Industry-Portfolios CSV -> loss matrix -> Frechet margins.
Run as:  python -m rmlm.data_prep
"""
import io
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "30_Industry_Portfolios_Daily.CSV"
OUT_LOSSES = Path(__file__).resolve().parents[2] / "data" / "processed" / "X_losses.csv"
OUT_FRECHET = Path(__file__).resolve().parents[2] / "data" / "processed" / "X_frechet.csv"

WINDOW_START = "1989-06-01"
WINDOW_END = "1998-06-15"


def load_raw(path=RAW_PATH):
    """
    Safely reads the Ken French CSV by extracting only the first table block
    (Value-Weighted) and ignoring header text and subsequent tables.
    """
    table_lines = []
    header_found = False
    
    # خواندن فایل سطر به سطر برای جداکردن فقط جدول اول
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            
            # پیدا کردن سطر هدر (که شامل اسامی صنایع است)
            if not header_found:
                if stripped.startswith(",Food") or ",Food" in stripped.replace(" ", ""):
                    header_found = True
                    table_lines.append(line)
                continue
            
            # به محض رسیدن به سطر خالی یا شروع جدول دوم، خواندن را متوقف کن
            if not stripped or "Equal Weighted" in line or "Copyright" in line:
                break
                
            table_lines.append(line)

    if not table_lines:
        raise ValueError("دیدن هدر جدول در فایل خام امکان‌پذیر نبود. لطفاً از وجود فایل مطمئن شوید.")

    # تبدیل متن جداشده به دیتای قابل خواندن توسط پانداس
    csv_data = "".join(table_lines)
    df = pd.read_csv(
        io.StringIO(csv_data),
        index_col=0,
        na_values=[-99.99, -999, -99.9900, -999.00]
    )
    
    # تمیزکاری نام ستون‌ها و ایندکس تاریخ
    df.columns = df.columns.str.strip()
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df[~df.index.isna()]
    df = df.apply(pd.to_numeric, errors="coerce")
    
    return df


def to_losses(df, window_start=WINDOW_START, window_end=WINDOW_END):
    window = df.loc[window_start:window_end]
    Xstar = window.values.astype(float) / 100.0
    X = np.maximum(-Xstar, 0.0)
    return X, list(window.columns), window.index


def to_frechet(X, alpha=2.0):
    n, d = X.shape
    U = np.zeros_like(X)
    for i in range(d):
        ranks = X[:, i].argsort().argsort() + 1
        U[:, i] = ranks / (n + 1)
    return (-np.log(U)) ** (-1.0 / alpha)


if __name__ == "__main__":
    df = load_raw()
    X, names, dates = to_losses(df)
    Xf = to_frechet(X)
    
    # ایجاد پوشه خروجی در صورت عدم وجود
    OUT_LOSSES.parent.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(X, columns=names, index=dates).to_csv(OUT_LOSSES)
    pd.DataFrame(Xf, columns=names, index=dates).to_csv(OUT_FRECHET)
    print(f"wrote {OUT_LOSSES} and {OUT_FRECHET}  shape={Xf.shape}")