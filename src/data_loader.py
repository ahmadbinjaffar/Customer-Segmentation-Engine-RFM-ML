"""
data_loader.py
==============
Fetches the UCI Online Retail II dataset and caches it locally as CSV.
Includes a robust fallback chain: ucimlrepo → direct URL → synthetic sample.

Usage:
    from src.data_loader import load_raw_data
    df = load_raw_data()
"""

import sys
import pandas as pd
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _generate_synthetic_dataset(n_customers: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Online Retail II dataset as a fallback.
    Preserves the exact column schema so all downstream modules work.
    """
    print("[data_loader] Generating synthetic dataset (UCI download unavailable)...")
    rng = np.random.default_rng(seed)

    countries = (
        ["United Kingdom"] * 80 + ["Germany"] * 4 + ["France"] * 4 +
        ["EIRE"] * 2 + ["Spain"] * 2 + ["Netherlands"] * 2 +
        ["Belgium", "Switzerland", "Portugal", "Australia", "Norway", "Italy"]
    )

    stock_codes = [f"{rng.integers(10000, 99999)}{chr(rng.integers(65, 91))}" for _ in range(500)]
    descriptions = [
        "WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN",
        "CREAM CUPID HEARTS COAT HANGER", "KNITTED UNION FLAG HOT WATER BOTTLE",
        "RED WOOLLY HOTTIE WHITE HEART", "SET 7 BABUSHKA NESTING BOXES",
        "GLASS STAR FROSTED T-LIGHT HOLDER", "HAND WARMER UNION JACK",
        "HAND WARMER RED POLKA DOT", "ASSORTED COLOUR BIRD ORNAMENT",
        "POPPY'S PLAYHOUSE BEDROOM", "POPPY'S PLAYHOUSE KITCHEN",
        "FELTCRAFT PRINCESS CHARLOTTE DOLL", "IVORY KNITTED MUG COSY",
        "BOX OF 6 ASSORTED COLOUR TEASPOONS", "LUNCH BAG RED RETROSPOT",
        "LUNCH BAG BLACK SKULL", "NATURAL SLATE HEART CHALKBOARD",
        "FAIRY CAKE FLANNEL ASSORTED COLOUR", "SPOTTY BUNTING",
    ]

    rows = []
    customer_ids = list(range(12346, 12346 + n_customers))
    base_date = pd.Timestamp("2009-12-01")
    end_date = pd.Timestamp("2011-12-09")

    for cust_id in customer_ids:
        n_invoices = rng.integers(1, 25)
        for inv_idx in range(n_invoices):
            inv_date = base_date + pd.Timedelta(days=int(rng.integers(0, (end_date - base_date).days)))
            invoice_no = f"{5 + inv_idx}{rng.integers(10000, 99999)}"
            n_items = rng.integers(1, 8)
            for _ in range(n_items):
                rows.append({
                    "Invoice": invoice_no,
                    "StockCode": rng.choice(stock_codes),
                    "Description": rng.choice(descriptions),
                    "Quantity": int(rng.integers(1, 25)),
                    "InvoiceDate": inv_date + pd.Timedelta(hours=int(rng.integers(7, 20)),
                                                           minutes=int(rng.integers(0, 60))),
                    "Price": round(float(rng.uniform(0.5, 25.0)), 2),
                    "Customer ID": float(cust_id),
                    "Country": rng.choice(countries),
                })

    # Inject ~5% cancelled invoices
    n_cancel = len(rows) // 20
    cancel_indices = rng.choice(len(rows), size=n_cancel, replace=False)
    for idx in cancel_indices:
        rows[idx]["Invoice"] = "C" + rows[idx]["Invoice"]
        rows[idx]["Quantity"] = -abs(rows[idx]["Quantity"])

    # Inject ~3% null customer IDs
    n_null = len(rows) // 33
    null_indices = rng.choice(len(rows), size=n_null, replace=False)
    for idx in null_indices:
        rows[idx]["Customer ID"] = np.nan

    df = pd.DataFrame(rows)
    print(f"[data_loader] Generated {len(df):,} synthetic transactions, {n_customers} customers")
    return df


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Online Retail II dataset.

    Priority chain:
    1. Cached CSV on disk
    2. ucimlrepo Python package
    3. Direct URL download
    4. Synthetic fallback dataset

    Parameters
    ----------
    force_download : bool
        If True, re-download even if a cached CSV exists.

    Returns
    -------
    pd.DataFrame
        Raw transaction data.
    """
    csv_path = config.RAW_DATA_DIR / config.RAW_CSV_FILENAME

    # ── 1. Cached CSV ────────────────────────────────────────────────────
    if csv_path.exists() and not force_download:
        print(f"[data_loader] Loading cached data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["InvoiceDate"])
        print(f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} cols")
        return df

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = None

    # ── 2. ucimlrepo package ─────────────────────────────────────────────
    try:
        print("[data_loader] Attempting download via ucimlrepo...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
        df = dataset.data.original
        print(f"[data_loader] Downloaded {len(df):,} rows via ucimlrepo")
    except Exception as e:
        print(f"[data_loader] ucimlrepo failed: {e}")

    # ── 3. Direct URL ────────────────────────────────────────────────────
    if df is None:
        try:
            print("[data_loader] Attempting direct URL download...")
            import urllib.request, ssl, zipfile, io
            ssl._create_default_https_context = ssl._create_unverified_context
            url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=90)
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            xlsx_names = [n for n in z.namelist() if n.endswith(".xlsx")]
            if xlsx_names:
                with z.open(xlsx_names[0]) as f:
                    df = pd.read_excel(f, engine="openpyxl")
                print(f"[data_loader] Downloaded {len(df):,} rows via direct URL")
        except Exception as e:
            print(f"[data_loader] Direct URL failed: {e}")

    # ── 4. Synthetic fallback ────────────────────────────────────────────
    if df is None:
        df = _generate_synthetic_dataset()

    # ── Cache locally ────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    print(f"[data_loader] Saved {len(df):,} rows to {csv_path}")
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """Return a summary dict describing the raw dataset for quality reporting."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "date_range": (
            str(df["InvoiceDate"].min()) if "InvoiceDate" in df.columns else None,
            str(df["InvoiceDate"].max()) if "InvoiceDate" in df.columns else None,
        ),
        "unique_customers": (
            df["Customer ID"].nunique() if "Customer ID" in df.columns else None
        ),
        "unique_invoices": (
            df["Invoice"].nunique() if "Invoice" in df.columns else None
        ),
        "countries": (
            df["Country"].nunique() if "Country" in df.columns else None
        ),
    }


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_raw_data()
    info = get_data_info(df)
    print("\n── Raw Data Summary ──")
    for k, v in info.items():
        print(f"  {k}: {v}")
