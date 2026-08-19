"""
data_cleaner.py
===============
Cleaning pipeline for the Online Retail II dataset.

Each cleaning step is a pure function that takes a DataFrame and returns
a cleaned DataFrame — making the pipeline composable and testable.

Usage:
    from src.data_cleaner import clean_pipeline
    df_clean = clean_pipeline(df_raw)
"""

import sys
import pandas as pd
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# =============================================================================
# Individual Cleaning Steps
# =============================================================================

def remove_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Customer ID is null (can't segment unknown customers)."""
    return df.dropna(subset=["Customer ID"]).copy()


def remove_cancelled_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled transactions.
    Cancelled invoices start with 'C' in the Invoice column.
    """
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    mask = ~df["Invoice"].str.startswith("C")
    return df[mask]


def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where Quantity > 0 and Price > 0."""
    return df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()


def add_total_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add TotalPrice = Quantity × Price column."""
    df = df.copy()
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure InvoiceDate is parsed as datetime."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def cast_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Customer ID to integer (removes .0 float artifacts)."""
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df


# =============================================================================
# Full Pipeline
# =============================================================================

CLEANING_STEPS = [
    ("Parse dates",              parse_dates),
    ("Remove missing customers", remove_missing_customers),
    ("Cast Customer ID",         cast_customer_id),
    ("Remove cancelled orders",  remove_cancelled_invoices),
    ("Remove negative values",   remove_negative_values),
    ("Add TotalPrice column",    add_total_price),
]


def clean_pipeline(
    df: pd.DataFrame,
    verbose: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline and optionally save the result.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data.
    verbose : bool
        Print row counts after each step.
    save : bool
        If True, save cleaned data to data/processed/.

    Returns
    -------
    pd.DataFrame
        Cleaned transaction data.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("  DATA CLEANING PIPELINE")
        print(f"{'='*60}")
        print(f"  Starting rows: {len(df):>10,}")

    for step_name, step_fn in CLEANING_STEPS:
        df = step_fn(df)
        if verbose:
            print(f"  After {step_name:<30s}: {len(df):>10,} rows")

    # Reset index
    df = df.reset_index(drop=True)

    if verbose:
        print(f"{'='*60}")
        print(f"  Final rows:    {len(df):>10,}")
        print(f"  Final columns: {list(df.columns)}")
        print(f"{'='*60}\n")

    # ── Save ─────────────────────────────────────────────────────────────
    if save:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.PROCESSED_DATA_DIR / config.CLEAN_CSV_FILENAME
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"  Saved cleaned data to {out_path}")

    return df


def get_cleaning_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """
    Produce a before/after quality summary for reporting.
    """
    return {
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean),
        "rows_removed": len(df_raw) - len(df_clean),
        "removal_pct": round((1 - len(df_clean) / len(df_raw)) * 100, 2),
        "clean_customers": df_clean["Customer ID"].nunique(),
        "clean_invoices": df_clean["Invoice"].nunique(),
        "clean_countries": df_clean["Country"].nunique(),
        "date_range": (
            str(df_clean["InvoiceDate"].min()),
            str(df_clean["InvoiceDate"].max()),
        ),
        "total_revenue": round(df_clean["TotalPrice"].sum(), 2),
        "avg_order_value": round(
            df_clean.groupby("Invoice")["TotalPrice"].sum().mean(), 2
        ),
    }


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_raw_data

    df_raw = load_raw_data()
    df_clean = clean_pipeline(df_raw, verbose=True, save=True)
    summary = get_cleaning_summary(df_raw, df_clean)
    print("\n── Cleaning Summary ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")
