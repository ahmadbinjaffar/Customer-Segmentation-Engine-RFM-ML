"""
clustering.py
Performs K-Means and DBSCAN clustering on RFM features.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def scale_features(rfm_df: pd.DataFrame, features: List[str] = ['Recency_log', 'Frequency_log', 'Monetary_log']) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale selected features using StandardScaler.
    
    Args:
        rfm_df: DataFrame containing the RFM data.
        features: List of feature columns to scale.
        
    Returns:
        Tuple containing the scaled array and the fitted scaler object.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(rfm_df[features])
    return scaled_array, scaler


def run_kmeans_elbow(X_scaled: np.ndarray, k_range=range(2, 11), random_state: int = 42) -> Dict[str, List[float]]:
    """
    Run K-Means for a range of k values to evaluate elbow metrics.
    
    Args:
        X_scaled: Scaled feature array.
        k_range: Range of k values to test.
        random_state: Random state for reproducibility.
        
    Returns:
        Dictionary containing inertias, silhouette scores, Calinski-Harabasz scores, and k_range.
    """
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    k_list = list(k_range)
    
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
        
    return {
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'k_range': k_list
    }


def find_optimal_k(elbow_results: Dict[str, List[float]]) -> int:
    """
    Find optimal k based on the highest silhouette score.
    
    Args:
        elbow_results: Results from run_kmeans_elbow.
        
    Returns:
        Optimal k value.
    """
    best_idx = np.argmax(elbow_results['silhouette_scores'])
    return elbow_results['k_range'][best_idx]


def run_kmeans(X_scaled: np.ndarray, n_clusters: int, random_state: int = 42) -> Dict[str, Any]:
    """
    Run K-Means clustering with the specified number of clusters.
    
    Args:
        X_scaled: Scaled feature array.
        n_clusters: Number of clusters.
        random_state: Random state.
        
    Returns:
        Dictionary with clustering results and evaluation metrics.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    
    return {
        'labels': labels,
        'model': kmeans,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
        'calinski': calinski_harabasz_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels)
    }


def run_dbscan_grid(X_scaled: np.ndarray, eps_candidates: List[float], min_samples_candidates: List[int]) -> List[Dict[str, Any]]:
    """
    Run DBSCAN clustering using a grid search of parameters.
    
    Args:
        X_scaled: Scaled feature array.
        eps_candidates: List of epsilon values to test.
        min_samples_candidates: List of min_samples values to test.
        
    Returns:
        List of dictionaries containing results for valid parameter combinations.
    """
    results = []
    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Skip configs that produce <2 clusters
            if n_clusters > 1:
                sil_score = silhouette_score(X_scaled, labels)
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': sil_score,
                    'labels': labels
                })
    return results


def find_best_dbscan(grid_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the best DBSCAN result based on silhouette score.
    
    Args:
        grid_results: List of valid DBSCAN results from run_dbscan_grid.
        
    Returns:
        Best result dictionary or None if grid_results is empty.
    """
    if not grid_results:
        return None
    return max(grid_results, key=lambda x: x['silhouette'])


def compare_methods(kmeans_result: Dict[str, Any], dbscan_result: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare K-Means and DBSCAN results.
    
    Args:
        kmeans_result: Best K-Means result.
        dbscan_result: Best DBSCAN result.
        
    Returns:
        DataFrame comparing the two methods.
    """
    comparison = {
        'Method': ['K-Means'],
        'Clusters': [len(set(kmeans_result['labels']))],
        'Noise Points': [0],
        'Silhouette Score': [kmeans_result['silhouette']]
    }
    
    if dbscan_result:
        comparison['Method'].append('DBSCAN')
        comparison['Clusters'].append(dbscan_result['n_clusters'])
        comparison['Noise Points'].append(dbscan_result['n_noise'])
        comparison['Silhouette Score'].append(dbscan_result['silhouette'])
        
    return pd.DataFrame(comparison)


def get_cluster_profiles(rfm_df: pd.DataFrame, labels: np.ndarray, method_name: str = 'KMeans') -> pd.DataFrame:
    """
    Calculate cluster-level statistics (mean R, F, M, count, pct).
    
    Args:
        rfm_df: Original RFM DataFrame.
        labels: Cluster labels.
        method_name: Name of the clustering method.
        
    Returns:
        DataFrame containing cluster profiles.
    """
    df = rfm_df.copy()
    df[f'{method_name}_Cluster'] = labels
    
    # Try to group by the standard CustomerID if present, otherwise assume rows are customers.
    # RFM DataFrames usually have one row per customer.
    agg_dict = {
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }
    
    if 'CustomerID' in df.columns:
        agg_dict['CustomerID'] = 'count'
        profile = df.groupby(f'{method_name}_Cluster').agg(agg_dict).rename(columns={'CustomerID': 'Count'})
    else:
        profile = df.groupby(f'{method_name}_Cluster').agg(agg_dict)
        profile['Count'] = df.groupby(f'{method_name}_Cluster').size()
        
    profile['Pct'] = profile['Count'] / profile['Count'].sum() * 100
    return profile


if __name__ == "__main__":
    # Use config directories or fallback to standard ones
    processed_dir = Path(getattr(config, 'PROCESSED_DATA_DIR', str(Path(__file__).resolve().parent.parent / 'data' / 'processed')))
    data_path = processed_dir / 'rfm_features.csv'
    
    # Try alternate filename if rfm_features.csv does not exist
    if not data_path.exists():
        data_path = processed_dir / 'rfm.csv'
        if not data_path.exists():
            print(f"Data file not found. Ensure processed data is present.")
            sys.exit(1)
        
    print(f"Loading data from {data_path}...")
    rfm_df = pd.read_csv(data_path)
    
    print("Scaling features...")
    # Assume default features exist. If they don't, scale what we can.
    log_features = ['Recency_log', 'Frequency_log', 'Monetary_log']
    if not all(col in rfm_df.columns for col in log_features):
        print(f"Missing some log-transformed features in the data. Adjusting feature list.")
        log_features = [col for col in rfm_df.columns if 'log' in col.lower() or col in ['Recency', 'Frequency', 'Monetary']]

    X_scaled, scaler = scale_features(rfm_df, features=log_features)
    
    # K-Means Analysis
    print("\n--- K-Means Clustering ---")
    k_range = getattr(config, 'K_RANGE', range(2, 11))
    random_state = getattr(config, 'RANDOM_STATE', 42)
    
    print(f"Testing K values in {list(k_range)}...")
    elbow_results = run_kmeans_elbow(X_scaled, k_range=k_range, random_state=random_state)
    optimal_k = find_optimal_k(elbow_results)
    print(f"Optimal K found based on silhouette score: {optimal_k}")
    
    kmeans_result = run_kmeans(X_scaled, n_clusters=optimal_k, random_state=random_state)
    print(f"K-Means Silhouette Score: {kmeans_result['silhouette']:.4f}")
    
    # DBSCAN Analysis
    print("\n--- DBSCAN Clustering ---")
    eps_candidates = getattr(config, 'DBSCAN_EPS_CANDIDATES', [0.3, 0.5, 0.7])
    min_samples_candidates = getattr(config, 'DBSCAN_MIN_SAMPLES_CANDIDATES', [5, 10, 15])
    
    print(f"Testing DBSCAN with eps {eps_candidates} and min_samples {min_samples_candidates}...")
    dbscan_grid = run_dbscan_grid(X_scaled, eps_candidates, min_samples_candidates)
    best_dbscan = find_best_dbscan(dbscan_grid)
    
    if best_dbscan:
        print(f"Best DBSCAN parameters: eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}")
        print(f"DBSCAN Silhouette Score: {best_dbscan['silhouette']:.4f}")
    else:
        print("No valid DBSCAN models found (all models produced <2 clusters or pure noise).")
    
    # Method Comparison
    print("\n--- Clustering Methods Comparison ---")
    comparison_df = compare_methods(kmeans_result, best_dbscan)
    print(comparison_df.to_string(index=False))
    
    # Cluster Profiles
    print("\n--- K-Means Cluster Profiles ---")
    kmeans_profiles = get_cluster_profiles(rfm_df, kmeans_result['labels'], method_name='KMeans')
    print(kmeans_profiles.round(2))
    
    if best_dbscan:
        print("\n--- DBSCAN Cluster Profiles ---")
        dbscan_profiles = get_cluster_profiles(rfm_df, best_dbscan['labels'], method_name='DBSCAN')
        print(dbscan_profiles.round(2))
