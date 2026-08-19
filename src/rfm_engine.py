import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import pandas as pd
import numpy as np

def compute_rfm(df: pd.DataFrame, analysis_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Computes RFM (Recency, Frequency, Monetary) features from cleaned transaction data.
    
    Args:
        df: DataFrame containing transaction data with columns 'Customer ID', 'InvoiceDate', 'Invoice', 'TotalPrice'
        analysis_date: Reference date for recency calculation. If None, max(InvoiceDate) + 1 day is used.
        
    Returns:
        DataFrame with columns: Customer ID, Recency, Frequency, Monetary
    """
    if analysis_date is None:
        analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
    rfm_df = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm_df.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    return rfm_df

def _safe_qcut(series: pd.Series, q: int, labels: list, ascending: bool = True) -> pd.Series:
    """Safely apply qcut with fallback to rank based cut if ties drop too many bins."""
    try:
        # Lower recency days is better, so if ascending=False for Recency, we want lower values to get higher scores.
        # Actually pd.qcut handles duplicates='drop' but labels length must match bins.
        # We can rank first to guarantee uniqueness if we want, or use duplicates='drop' and adjust labels.
        # Standard approach for RFM with ties: rank with method='first'
        if not ascending:
            ranked = series.rank(method='first', ascending=False)
        else:
            ranked = series.rank(method='first', ascending=True)
            
        return pd.qcut(ranked, q=q, labels=labels)
    except Exception:
        # Fallback if needed, though rank(method='first') guarantees unique values
        return pd.qcut(series, q=q, labels=labels, duplicates='drop')

def add_rfm_scores(rfm_df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """
    Adds R_Score, F_Score, M_Score, and RFM_Score columns.
    Higher R_Score = more recent. Higher F/M_Score = more frequent/higher spend.
    """
    df = rfm_df.copy()
    
    r_labels = list(range(quantiles, 0, -1)) # 5 to 1 (lower recency days gets higher score)
    f_labels = list(range(1, quantiles + 1)) # 1 to 5
    m_labels = list(range(1, quantiles + 1)) # 1 to 5
    
    # Actually wait, if we rank ascending=False (higher recency days -> lower rank) and qcut labels 1 to 5.
    # Let's just rank ascending=True and apply reverse labels for Recency.
    df['R_Score'] = pd.qcut(df['Recency'].rank(method='first'), q=quantiles, labels=r_labels).astype(int)
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), q=quantiles, labels=f_labels).astype(int)
    df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), q=quantiles, labels=m_labels).astype(int)
    
    df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    
    return df

def add_rfm_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'Segment' column mapping RFM scores to 11 business segments based on R_Score and FM_Score.
    """
    df = rfm_df.copy()
    
    # Calculate FM_Score average
    df['FM_Score'] = ((df['F_Score'] + df['M_Score']) / 2).round().astype(int)
    
    def map_segment(row):
        r = row['R_Score']
        fm = row['FM_Score']
        
        if r >= 4 and fm >= 4:
            return 'Champions'
        elif r >= 3 and fm >= 3:
            return 'Loyal Customers'
        elif r >= 4 and fm >= 2:
            return 'Potential Loyalists'
        elif r >= 4 and fm == 1:
            return 'Recent Customers'
        elif r == 3 and fm == 2:
            return 'Promising'
        elif r == 3 and fm == 1:
            return 'About to Sleep'
        elif r == 2 and fm >= 3:
            return 'Need Attention'
        elif r <= 2 and fm >= 4:
            return "Can't Lose"
        elif r == 2 and fm <= 2:
            return 'Hibernating'
        elif r == 1 and fm == 3:
            return 'At Risk'
        elif r == 1 and fm <= 2:
            return 'Lost'
        else:
            return 'Other'
            
    df['Segment'] = df.apply(map_segment, axis=1)
    
    return df

def log_transform_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with log transformed Recency, Frequency, and Monetary columns (using np.log1p)."""
    df = rfm_df.copy()
    # Handle negative Monetary values just in case
    df['Monetary'] = df['Monetary'].clip(lower=0)
    
    df['Recency_log'] = np.log1p(df['Recency'])
    df['Frequency_log'] = np.log1p(df['Frequency'])
    df['Monetary_log'] = np.log1p(df['Monetary'])
    
    return df

def get_rfm_summary(rfm_df: pd.DataFrame) -> dict:
    """Returns a dictionary containing summary statistics of the RFM features."""
    summary = {
        'num_customers': len(rfm_df),
        'recency_mean': rfm_df['Recency'].mean(),
        'recency_median': rfm_df['Recency'].median(),
        'frequency_mean': rfm_df['Frequency'].mean(),
        'frequency_median': rfm_df['Frequency'].median(),
        'monetary_mean': rfm_df['Monetary'].mean(),
        'monetary_median': rfm_df['Monetary'].median()
    }
    
    if 'Segment' in rfm_df.columns:
        summary['segment_counts'] = rfm_df['Segment'].value_counts().to_dict()
        
    return summary

if __name__ == "__main__":
    print("Loading cleaned transaction data...")
    data_path = Path(config.PROCESSED_DATA_DIR) / "clean_transactions.csv"
    
    if not data_path.exists():
        print(f"Error: Could not find data file at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, parse_dates=['InvoiceDate'])
    
    print("Computing RFM features...")
    analysis_date = getattr(config, 'ANALYSIS_DATE', None)
    if analysis_date is not None:
        analysis_date = pd.to_datetime(analysis_date)
        
    rfm = compute_rfm(df, analysis_date)
    
    print("Adding RFM scores...")
    quantiles = getattr(config, 'RFM_QUANTILES', 5)
    rfm = add_rfm_scores(rfm, quantiles)
    
    print("Adding RFM segments...")
    rfm = add_rfm_segments(rfm)
    
    print("Applying log transformations...")
    rfm = log_transform_rfm(rfm)
    
    print("\nRFM Summary:")
    summary = get_rfm_summary(rfm)
    for k, v in summary.items():
        if k == 'segment_counts':
            print(f"\n{k}:")
            for seg, count in v.items():
                print(f"  {seg}: {count}")
        else:
            print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
            
    out_path = Path(config.PROCESSED_DATA_DIR) / "rfm_features.csv"
    rfm.to_csv(out_path, index=False)
    print(f"\nSaved RFM features to {out_path}")
