"""Tests for src/rfm_engine.py — uses synthetic data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.rfm_engine import compute_rfm, add_rfm_scores, add_rfm_segments, log_transform_rfm


@pytest.fixture
def clean_transactions():
    """Synthetic clean transaction data with 5 customers and varied patterns."""
    np.random.seed(42)
    rows = []
    for cust_id in range(1, 6):
        n_orders = np.random.randint(2, 8)
        for i in range(n_orders):
            rows.append({
                "Customer ID": cust_id,
                "Invoice": f"INV-{cust_id}-{i}",
                "InvoiceDate": pd.Timestamp("2011-12-01") - pd.Timedelta(days=int(np.random.randint(1, 365))),
                "TotalPrice": round(np.random.uniform(5.0, 500.0), 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def rfm_table(clean_transactions):
    """Pre-computed RFM table from synthetic data."""
    return compute_rfm(clean_transactions, pd.Timestamp("2011-12-10"))


def test_compute_rfm_shape(clean_transactions):
    rfm = compute_rfm(clean_transactions, pd.Timestamp("2011-12-10"))
    assert len(rfm) == 5  # 5 unique customers
    assert "Customer ID" in rfm.columns
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert rfm["Customer ID"].nunique() == len(rfm)


def test_compute_rfm_values(rfm_table):
    assert (rfm_table["Recency"] >= 0).all()
    assert (rfm_table["Frequency"] >= 1).all()
    assert (rfm_table["Monetary"] > 0).all()


def test_add_rfm_scores_range(rfm_table):
    scored = add_rfm_scores(rfm_table, quantiles=5)
    for col in ["R_Score", "F_Score", "M_Score"]:
        assert col in scored.columns
        assert scored[col].isin([1, 2, 3, 4, 5]).all()
    assert "RFM_Score" in scored.columns


def test_add_rfm_segments(rfm_table):
    scored = add_rfm_scores(rfm_table)
    segmented = add_rfm_segments(scored)
    assert "Segment" in segmented.columns
    # All segments should be one of the 11 defined + possible "Other"
    valid = {
        "Champions", "Loyal Customers", "Potential Loyalists",
        "Recent Customers", "Promising", "Need Attention",
        "About to Sleep", "At Risk", "Can't Lose",
        "Hibernating", "Lost", "Other",
    }
    assert set(segmented["Segment"].unique()).issubset(valid)


def test_log_transform(rfm_table):
    log_df = log_transform_rfm(rfm_table)
    for col in ["Recency_log", "Frequency_log", "Monetary_log"]:
        assert col in log_df.columns
        assert np.isfinite(log_df[col]).all()
