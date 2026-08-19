"""Run the full data pipeline end-to-end and print summary."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import load_raw_data, get_data_info
from src.data_cleaner import clean_pipeline, get_cleaning_summary
from src.rfm_engine import compute_rfm, add_rfm_scores, add_rfm_segments, log_transform_rfm, get_rfm_summary
from src.clustering import (
    scale_features, run_kmeans_elbow, find_optimal_k,
    run_kmeans, run_dbscan_grid, find_best_dbscan, compare_methods,
)


def main():
    # ── Load ─────────────────────────────────────────────────────────────
    df_raw = load_raw_data()
    info = get_data_info(df_raw)
    print("\n=== RAW DATA ===")
    print(f"  Shape: {info['shape']}")
    print(f"  Date range: {info['date_range']}")
    print(f"  Unique customers: {info['unique_customers']}")
    print(f"  Null %: Customer ID={info['null_pct'].get('Customer ID', 'N/A')}%")

    # ── Clean ────────────────────────────────────────────────────────────
    df_clean = clean_pipeline(df_raw, verbose=True, save=True)
    summary = get_cleaning_summary(df_raw, df_clean)
    print("\n=== CLEANING SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ── RFM ──────────────────────────────────────────────────────────────
    rfm = compute_rfm(df_clean)
    rfm = add_rfm_scores(rfm)
    rfm = add_rfm_segments(rfm)
    rfm = log_transform_rfm(rfm)
    rfm_summary = get_rfm_summary(rfm)
    print("\n=== RFM SUMMARY ===")
    for k, v in rfm_summary.items():
        if k == "segment_counts":
            print("  Segments:")
            for seg, cnt in v.items():
                print(f"    {seg}: {cnt}")
        else:
            fmt = f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}"
            print(fmt)

    # Save
    rfm.to_csv(config.PROCESSED_DATA_DIR / config.SEGMENTS_CSV_FILENAME, index=False)
    print(f"\n  Saved segments -> {config.PROCESSED_DATA_DIR / config.SEGMENTS_CSV_FILENAME}")

    # ── Clustering ───────────────────────────────────────────────────────
    X_scaled, scaler = scale_features(rfm)
    elbow = run_kmeans_elbow(X_scaled, config.K_RANGE, config.RANDOM_STATE)
    optimal_k = find_optimal_k(elbow)
    print(f"\n=== CLUSTERING ===")
    print(f"  Optimal K (silhouette): {optimal_k}")

    kmeans_result = run_kmeans(X_scaled, optimal_k, config.RANDOM_STATE)
    print(f"  K-Means silhouette: {kmeans_result['silhouette']:.4f}")
    print(f"  K-Means Calinski-Harabasz: {kmeans_result['calinski']:.1f}")
    print(f"  K-Means Davies-Bouldin: {kmeans_result['davies_bouldin']:.4f}")

    dbscan_grid = run_dbscan_grid(
        X_scaled, config.DBSCAN_EPS_CANDIDATES, config.DBSCAN_MIN_SAMPLES_CANDIDATES
    )
    best_dbscan = find_best_dbscan(dbscan_grid)
    if best_dbscan:
        print(f"  DBSCAN best: eps={best_dbscan['eps']}, "
              f"min_samples={best_dbscan['min_samples']}, "
              f"silhouette={best_dbscan['silhouette']:.4f}, "
              f"clusters={best_dbscan['n_clusters']}, noise={best_dbscan['n_noise']}")
    else:
        print("  DBSCAN: No valid configuration found")

    comparison = compare_methods(kmeans_result, best_dbscan)
    print("\n=== METHOD COMPARISON ===")
    print(comparison.to_string(index=False))

    # Save a sample output
    rfm_out = rfm.copy()
    rfm_out["KMeans_Cluster"] = kmeans_result["labels"]
    sample_path = config.OUTPUT_DIR / "rfm_segments_sample.csv"
    rfm_out.head(50).to_csv(sample_path, index=False)
    print(f"\n  Sample output -> {sample_path}")

    print("\n[OK] Full pipeline completed successfully!")


if __name__ == "__main__":
    main()
