# 🎯 Customer Segmentation Engine (RFM + ML Clustering)

## 📌 Executive Summary
This project is an **end-to-end Machine Learning and Business Intelligence solution** designed to segment e-commerce customers based on purchasing behavior. 

Using **525,461 raw transactions** from the **UCI Online Retail II dataset** (spanning 37 countries), the system applies:
1. A **6-step data cleaning and validation pipeline**
2. **RFM (Recency, Frequency, Monetary) Feature Engineering** with quantile scoring into **11 human-interpretable business segments**
3. **Unsupervised Machine Learning** using **K-Means** (elbow & silhouette optimization) and **DBSCAN** (density grid search)
4. An **interactive 5-tab Streamlit web application** with 3D visualizations, segment profiling, customer lookup, and automated marketing recommendations.

---

## 💡 Problem Statement & Business Context

### The Challenge
E-commerce and retail businesses often treat all customers uniformly, leading to ineffective marketing campaigns, high customer acquisition costs, and customer churn. 

### The Solution
Customer segmentation allows businesses to target high-value customers, re-engage churned buyers, and optimize marketing spend. This system transforms raw transactional logs (`Invoice`, `StockCode`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`) into actionable business segments.

---

## 📊 Dataset Overview

* **Source**: [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
* **Dataset Period**: December 1, 2009 – December 9, 2010
* **Initial Records**: 525,461 transaction rows
* **Unique Customers**: 4,383
* **Target Audience**: Wholesalers and individual retail shoppers across 37 countries

---

## 🧹 Data Cleaning Pipeline

Raw retail data contains noise, missing values, cancellations, and data entry errors. The pipeline applies a strict 6-step cleaning process:

| Step | Operation | Rationale | Impact |
| :--- | :--- | :--- | :--- |
| **1** | Date Parsing | Convert `InvoiceDate` to standard `datetime64[ns]` | Enables exact recency calculations |
| **2** | Filter Missing Customer IDs | Drop records where `Customer ID` is null (20.54% of raw data) | Anonymous transactions cannot be tracked per customer |
| **3** | Customer ID Type Casting | Cast `Customer ID` from float to `int` | Clean identifier format |
| **4** | Remove Cancellations | Remove invoices starting with `'C'` | Prevents negative transaction counts |
| **5** | Filter Non-Positive Values | Keep only `Quantity > 0` and `Price > 0` | Filters out stock adjustments and data entry errors |
| **6** | Feature Engineering | Compute TotalPrice = Quantity * Price | Generates item transaction revenue |

### Cleaning Summary
* **Raw Rows**: 525,461
* **Clean Rows**: 407,664 (**22.42% removed**)
* **Cleaned Customers**: **4,312**
* **Total Cleaned Revenue**: **$8,832,003.27**
* **Average Order Value (AOV)**: **$459.69**

---

## 🧮 RFM Feature Engineering & Scoring

RFM measures three core attributes per customer relative to the dataset snapshot date (Max Invoice Date + 1 day):

* **Recency (R)** = Analysis Date - Latest Purchase Date (in days)
* **Frequency (F)** = Count of Unique Invoices per Customer
* **Monetary (M)** = Sum of TotalPrice per Customer

### Quantile Scoring (Quintiles 1–5)
Each customer receives a score from 1 to 5 for Recency, Frequency, and Monetary using pd.qcut:
* **R Score**: 5 = Most Recent (lowest recency days), 1 = Least Recent
* **F Score**: 5 = Highest Frequency, 1 = Lowest Frequency
* **M Score**: 5 = Highest Spend, 1 = Lowest Spend

---

## 🏷️ Rule-Based Business Segments (11 Categories)

Combining the R Score and FM Score (Average of F and M), customers map into 11 distinct business personas:

| Segment | Customers | % | Characteristics | Strategy / Recommendation |
| :--- | :---: | :---: | :--- | :--- |
| **Champions** | 1,135 | 26.3% | High R, F, & M scores. Bought recently, buy often, spend the most. | Reward them. Early product releases & VIP programs. |
| **Lost** | 691 | 16.0% | Low R, F, & M scores. Haven't bought in a long time. | Win-back offers or survey to understand why they left. |
| **Loyal Customers** | 645 | 15.0% | High F & M scores, moderate to high R. Buy regularly. | Upsell higher-value products. Ask for reviews. |
| **Hibernating** | 441 | 10.2% | Low R & F, low to moderate M. Inactive for months. | Re-activate with relevant steep discount offers. |
| **Need Attention** | 421 | 9.8% | Above average R, F, M. Recent activity dropping. | Limited-time deals, personalized recommendations. |
| **Potential Loyalists** | 368 | 8.5% | Recent buyers with average frequency. | Offer loyalty programs, recommend related products. |
| **Promising** | 310 | 7.2% | Recent buyers, low frequency, moderate spend. | Onboarding emails, discount on 2nd purchase. |
| **Can't Lose** | 91 | 2.1% | High F & M, but very low R. Big spenders who stopped buying. | Dedicated customer service calls, high-value win-back. |
| **At Risk** | 81 | 1.9% | High F & M in past, but haven't purchased in a long time. | Renewal notifications, aggressive discount campaigns. |
| **About to Sleep** | 71 | 1.6% | Below average R, F, M. Risk of losing them. | Share popular products and personalized offers. |
| **Recent Customers** | 58 | 1.3% | Bought recently, but low total transaction count. | Welcome series, educate on product catalog. |

---

## 🤖 Unsupervised Machine Learning (Clustering)

To complement rule-based RFM scoring, machine learning identifies natural mathematical groupings in continuous 3D RFM space.

### Preprocessing
1. **Log Transformation**: Metric_log = ln(1 + Metric) to reduce heavy right-skewness.
2. **Standardization**: `StandardScaler()` to scale features to mean = 0 and variance = 1.

### Algorithm 1: K-Means Clustering
* **Optimization**: Tested k in range [2, 10] via Elbow Method & Silhouette Analysis.
* **Optimal Result**: k = 2
* **Silhouette Score**: **0.4221**
* **Calinski-Harabasz Index**: **4,132.7**
* **Davies-Bouldin Index**: **0.9112**

### Algorithm 2: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
* **Optimization**: Grid search over eps in range [0.3, 1.5] and min_samples in range [3, 15].
* **Optimal Result**: eps = 0.7, min_samples = 5
* **Silhouette Score**: **0.5320**
* **Clusters Found**: 2 core clusters + 22 noise/outlier points.

### Clustering Takeaway
Both algorithms converged on **2 primary macro-clusters** (High-Value Power Users vs. Low-Value Occasional Buyers), while DBSCAN achieved a higher silhouette score (`0.5320`) by detecting dense core clusters and isolating noise points.

---

## 💻 Tech Stack & Architecture

```
                                  ┌────────────────────────┐
                                  │ UCI Online Retail II   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
┌─────────────────────────┐       ┌────────────────────────┐
│   pytest Test Suite     │◄──────┤   src/data_loader.py   │
│   (16/16 Unit Tests)    │       └───────────┬────────────┘
└─────────────────────────┘                   │
                                              ▼
                                  ┌────────────────────────┐
                                  │  src/data_cleaner.py   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   src/rfm_engine.py    │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   src/clustering.py    │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  app/streamlit_app.py  │
                                  │  (5-Tab Dashboard)     │
                                  └───────────┬────────────┘
```

### Technology Matrix
* **Language**: Python 3.10+
* **Data Processing**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn` (`KMeans`, `DBSCAN`, `StandardScaler`, metrics)
* **Visualizations**: `plotly` (3D scatter plots, heatmaps, treemaps, radar charts), `matplotlib`, `seaborn`
* **Dashboard Framework**: `streamlit`
* **Unit Testing**: `pytest`

---

## 📁 Repository Structure

```text
customer-segmentation-rfm/
├── .gitignore
├── .streamlit/
│   └── config.toml                  # Dark theme styling configuration
├── LICENSE                          # MIT License
├── README.md                        # Portfolio documentation
├── PROJECT_DOCUMENTATION.md         # Full technical documentation
├── requirements.txt                 # Project dependencies
├── config.py                        # Centralized configuration & parameters
├── app/
│   └── streamlit_app.py             # 5-Tab Streamlit web app
├── data/                            (git-ignored)
│   ├── raw/                         # Raw dataset storage
│   └── processed/                   # Cleaned data & RFM outputs
├── notebooks/
│   └── 01_eda_rfm_clustering.ipynb # Exploratory data analysis notebook
├── outputs/
│   ├── figures/                     # Saved charts & plots
│   └── rfm_segments_sample.csv      # Sample output dataset
├── scripts/
│   └── run_pipeline.py              # CLI entry point to run full pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading with 3-tier fallback
│   ├── data_cleaner.py              # 6-step cleaning pipeline
│   ├── rfm_engine.py                # RFM scoring & segmentation engine
│   ├── clustering.py                # K-Means & DBSCAN algorithms
│   ├── segment_labels.py            # Business segment definitions & actions
│   └── visualizations.py            # Plotly visualization suite
└── tests/
    ├── __init__.py
    ├── test_data_cleaner.py         # Unit tests for cleaning pipeline
    ├── test_rfm_engine.py           # Unit tests for RFM calculation
    └── test_clustering.py           # Unit tests for ML clustering
```

---

## 📝 Resume & Portfolio Bullet Points

### Resume Bullet
> *Built an end-to-end customer segmentation system using RFM analysis and K-Means/DBSCAN clustering on 525K retail transactions (4,312 customers across 37 countries), achieving a 0.53 silhouette score. Deployed an interactive 5-tab Streamlit dashboard with 11 actionable business segments and tailored marketing recommendations.*

### Short Portfolio / LinkedIn Project Description
> *Customer Segmentation Engine that combines RFM (Recency, Frequency, Monetary) feature engineering with unsupervised machine learning (K-Means and DBSCAN) to classify 4,312 retail customers into 11 actionable business segments. The system processes 525K transactions from the UCI Online Retail II dataset through a modular Python pipeline backed by 16 unit tests. Features an interactive Streamlit dashboard with glassmorphism UI, 3D cluster visualizations, customer lookup tool, and automated marketing strategies.*
