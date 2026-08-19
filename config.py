"""
config.py
=========
Central configuration for the Customer Segmentation project.
All tuneable constants live here — no magic numbers scattered in code.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# =============================================================================
# Dataset
# =============================================================================
UCI_DATASET_ID = 502                       # Online Retail II
RAW_CSV_FILENAME = "online_retail_ii.csv"
CLEAN_CSV_FILENAME = "clean_transactions.csv"
RFM_CSV_FILENAME = "rfm_table.csv"
SEGMENTS_CSV_FILENAME = "rfm_segments.csv"

# =============================================================================
# RFM Analysis
# =============================================================================
# Recency is computed relative to the day AFTER the last transaction in the dataset.
# Set to None to auto-detect from data (max InvoiceDate + 1 day).
ANALYSIS_DATE = None

# Number of quantile bins for R, F, M scoring
RFM_QUANTILES = 5

# =============================================================================
# Clustering
# =============================================================================
K_RANGE = range(2, 11)                     # k values to try for K-Means elbow
RANDOM_STATE = 42                          # reproducibility seed
DBSCAN_EPS_CANDIDATES = [0.3, 0.5, 0.7, 1.0, 1.5]
DBSCAN_MIN_SAMPLES_CANDIDATES = [3, 5, 10, 15]

# =============================================================================
# Streamlit Theme
# =============================================================================
PRIMARY_COLOR = "#6C63FF"
BACKGROUND_COLOR = "#0E1117"
SECONDARY_BG = "#1A1D29"
TEXT_COLOR = "#FAFAFA"
