"""Tests for src/clustering.py — uses synthetic scaled data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.clustering import (
    scale_features,
    run_kmeans_elbow,
    find_optimal_k,
    run_kmeans,
    run_dbscan_grid,
    get_cluster_profiles,
)


@pytest.fixture
def rfm_with_logs():
    """Synthetic RFM data with log-transformed columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Recency": np.random.randint(1, 400, n),
        "Frequency": np.random.randint(1, 50, n),
        "Monetary": np.random.uniform(10, 5000, n),
        "Recency_log": np.random.normal(3, 1, n),
        "Frequency_log": np.random.normal(1.5, 0.5, n),
        "Monetary_log": np.random.normal(5, 1.2, n),
    })


@pytest.fixture
def X_scaled(rfm_with_logs):
    """Scaled feature array from rfm_with_logs."""
    scaled, _ = scale_features(rfm_with_logs)
    return scaled


def test_scale_features(rfm_with_logs):
    X, scaler = scale_features(rfm_with_logs)
    assert X.shape == (len(rfm_with_logs), 3)
    assert hasattr(scaler, "mean_")
    # Scaled data should have ~zero mean
    assert abs(X.mean(axis=0)).max() < 0.1


def test_kmeans_elbow(X_scaled):
    k_range = range(2, 6)
    results = run_kmeans_elbow(X_scaled, k_range)
    assert "k_range" in results
    assert "inertias" in results
    assert "silhouette_scores" in results
    assert "calinski_scores" in results
    assert len(results["inertias"]) == 4
    assert len(results["silhouette_scores"]) == 4


def test_find_optimal_k(X_scaled):
    results = run_kmeans_elbow(X_scaled, range(2, 6))
    optimal = find_optimal_k(results)
    assert isinstance(optimal, (int, np.integer))
    assert 2 <= optimal <= 5


def test_run_kmeans(X_scaled):
    result = run_kmeans(X_scaled, n_clusters=3)
    assert len(result["labels"]) == X_scaled.shape[0]
    assert isinstance(result["silhouette"], float)
    assert isinstance(result["calinski"], float)
    assert isinstance(result["davies_bouldin"], float)
    assert result["silhouette"] > -1 and result["silhouette"] <= 1


def test_dbscan_grid(X_scaled):
    results = run_dbscan_grid(X_scaled, eps_candidates=[0.5, 1.0, 1.5], min_samples_candidates=[3, 5])
    assert isinstance(results, list)
    for r in results:
        assert "eps" in r
        assert "min_samples" in r
        assert "n_clusters" in r
        assert r["n_clusters"] >= 2


def test_cluster_profiles(rfm_with_logs, X_scaled):
    result = run_kmeans(X_scaled, n_clusters=3)
    profiles = get_cluster_profiles(rfm_with_logs, result["labels"], "KMeans")
    assert isinstance(profiles, pd.DataFrame)
    assert "Count" in profiles.columns
    assert "Pct" in profiles.columns
    assert "Recency" in profiles.columns
    assert profiles["Count"].sum() == len(rfm_with_logs)
