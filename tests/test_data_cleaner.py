"""Tests for src/data_cleaner.py — uses synthetic data, no network calls."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.data_cleaner import (
    remove_missing_customers,
    remove_cancelled_invoices,
    remove_negative_values,
    add_total_price,
    clean_pipeline,
)


@pytest.fixture
def sample_df():
    """Synthetic 'dirty' transaction data with nulls, cancellations, negatives."""
    data = {
        "Invoice": ["536365", "C536366", "536367", "536368", "536369", "536370", "536371"],
        "StockCode": ["85123A", "71053", "84406B", "84029G", "84029E", "22752", "21730"],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "CREAM CUPID HEARTS COAT HANGER",
            "KNITTED UNION FLAG HOT WATER BOTTLE",
            "RED WOOLLY HOTTIE WHITE HEART.",
            "SET 7 BABUSHKA NESTING BOXES",
            "GLASS STAR FROSTED T-LIGHT HOLDER",
        ],
        "Quantity": [6, -2, 8, -5, 6, 2, 6],
        "InvoiceDate": pd.to_datetime([
            "2010-12-01 08:26:00", "2010-12-01 08:28:00",
            "2010-12-01 08:34:00", "2010-12-01 08:34:00",
            "2010-12-01 08:35:00", "2010-12-01 08:45:00",
            "2010-12-01 09:00:00",
        ]),
        "Price": [2.55, 3.39, 2.75, 3.39, 3.39, 7.65, 4.25],
        "Customer ID": [17850.0, 17850.0, np.nan, 13047.0, 13047.0, 12583.0, 13047.0],
        "Country": ["United Kingdom"] * 5 + ["France", "United Kingdom"],
    }
    return pd.DataFrame(data)


def test_remove_missing_customers(sample_df):
    result = remove_missing_customers(sample_df)
    assert result["Customer ID"].isnull().sum() == 0
    assert len(result) == 6  # 1 null row removed


def test_remove_cancelled_invoices(sample_df):
    result = remove_cancelled_invoices(sample_df)
    assert not result["Invoice"].str.startswith("C").any()
    assert len(result) == 6  # 1 cancelled removed


def test_remove_negative_values(sample_df):
    result = remove_negative_values(sample_df)
    assert (result["Quantity"] > 0).all()
    assert (result["Price"] > 0).all()
    assert len(result) == 5  # 2 negative-qty rows removed


def test_add_total_price(sample_df):
    result = add_total_price(sample_df)
    assert "TotalPrice" in result.columns
    expected = sample_df["Quantity"] * sample_df["Price"]
    pd.testing.assert_series_equal(result["TotalPrice"], expected, check_names=False)


def test_clean_pipeline(sample_df):
    result = clean_pipeline(sample_df, verbose=False, save=False)
    assert result["Customer ID"].isnull().sum() == 0
    assert not result["Invoice"].str.startswith("C").any()
    assert (result["Quantity"] > 0).all()
    assert (result["Price"] > 0).all()
    assert "TotalPrice" in result.columns
    # 7 rows → remove: null CustID (1), cancelled+neg (1), neg-qty-only (1) = 4 remain
    assert len(result) == 4
