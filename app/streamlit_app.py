"""
streamlit_app.py
================
Premium Customer Segmentation Dashboard.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

# Set page config FIRST — must be the first Streamlit command
st.set_page_config(
    page_title="Customer Segmentation | RFM + Clustering",
    page_icon="🎯",
    layout="wide",
)

import config
from src.visualizations import (
    plot_rfm_distributions, plot_rfm_3d_scatter, plot_elbow,
    plot_silhouette_comparison, plot_segment_treemap, plot_segment_bar,
    plot_cluster_radar, plot_rfm_heatmap, plot_snake_plot, plot_revenue_by_segment,
)
from src.segment_labels import (
    SEGMENT_CONFIG, get_segment_summary, get_marketing_actions,
)


# ═════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═════════════════════════════════════════════════════════════════════════════
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-header {
        background: linear-gradient(135deg, #0E1117 0%, #1A1D29 50%, #0E1117 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }

    .hero-title {
        background: linear-gradient(135deg, #6C63FF, #4ECDC4, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.8rem;
        text-align: center;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #A0AEC0;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(108, 99, 255, 0.15);
        border-color: rgba(108, 99, 255, 0.3);
    }

    .metric-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .segment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    .action-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }

    .action-card:hover {
        transform: translateX(5px);
    }

    /* Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(108, 99, 255, 0.1);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Cached Data Pipeline
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_process():
    """Full data pipeline: load → clean → RFM → score → segment → log-transform."""
    from src.data_loader import load_raw_data
    from src.data_cleaner import clean_pipeline
    from src.rfm_engine import compute_rfm, add_rfm_scores, add_rfm_segments, log_transform_rfm

    df_raw = load_raw_data()
    df_clean = clean_pipeline(df_raw, verbose=False, save=False)
    rfm = compute_rfm(df_clean)
    rfm = add_rfm_scores(rfm)
    rfm = add_rfm_segments(rfm)
    rfm = log_transform_rfm(rfm)
    return df_raw, df_clean, rfm


@st.cache_data(show_spinner=False)
def run_clustering_pipeline(_rfm_df):
    """Run K-Means and DBSCAN clustering."""
    from src.clustering import (
        scale_features, run_kmeans_elbow, find_optimal_k,
        run_kmeans, run_dbscan_grid, find_best_dbscan,
        compare_methods, get_cluster_profiles,
    )

    X_scaled, scaler = scale_features(_rfm_df)
    elbow = run_kmeans_elbow(X_scaled, config.K_RANGE, config.RANDOM_STATE)
    optimal_k = find_optimal_k(elbow)
    kmeans_result = run_kmeans(X_scaled, optimal_k, config.RANDOM_STATE)
    dbscan_grid = run_dbscan_grid(
        X_scaled,
        config.DBSCAN_EPS_CANDIDATES,
        config.DBSCAN_MIN_SAMPLES_CANDIDATES,
    )
    best_dbscan = find_best_dbscan(dbscan_grid)
    comparison = compare_methods(kmeans_result, best_dbscan)
    kmeans_profiles = get_cluster_profiles(_rfm_df, kmeans_result["labels"], "KMeans")
    return X_scaled, elbow, optimal_k, kmeans_result, best_dbscan, comparison, kmeans_profiles


# ═════════════════════════════════════════════════════════════════════════════
# Main App
# ═════════════════════════════════════════════════════════════════════════════

def main():
    inject_custom_css()

    # ── Hero Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">🎯 Customer Segmentation Dashboard</p>
        <p class="hero-subtitle">RFM Analysis × K-Means & DBSCAN Clustering — Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ────────────────────────────────────────────────────────
    with st.spinner("🔄 Loading and processing data..."):
        df_raw, df_clean, rfm = load_and_process()
        (X_scaled, elbow, optimal_k, kmeans_result,
         best_dbscan, comparison_df, kmeans_profiles) = run_clustering_pipeline(rfm)

    # ── Compute KPIs ─────────────────────────────────────────────────────
    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_order_value = (
        df_clean.groupby("Invoice")["TotalPrice"].sum().mean()
        if "TotalPrice" in df_clean.columns
        else total_revenue / rfm["Frequency"].sum()
    )
    unique_countries = df_clean["Country"].nunique() if "Country" in df_clean.columns else "N/A"
    date_min = df_clean["InvoiceDate"].min().strftime("%b %Y") if "InvoiceDate" in df_clean.columns else "N/A"
    date_max = df_clean["InvoiceDate"].max().strftime("%b %Y") if "InvoiceDate" in df_clean.columns else "N/A"

    # ── KPI Cards ────────────────────────────────────────────────────────
    kpi_cols = st.columns(5)
    kpi_data = [
        ("Total Customers", f"{total_customers:,}"),
        ("Total Revenue", f"${total_revenue:,.0f}"),
        ("Avg Order Value", f"${avg_order_value:,.2f}"),
        ("Countries", f"{unique_countries}"),
        ("Date Range", f"{date_min} — {date_max}"),
    ]
    for col, (label, value) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview",
        "📊 RFM Analysis",
        "🤖 Clustering",
        "🔍 Customer Lookup",
        "📋 Segment Profiles",
    ])

    # ── Tab 1: Overview ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Welcome to the Customer Intelligence Portal")
        st.markdown(
            "This dashboard leverages **Recency, Frequency, and Monetary (RFM)** "
            "analysis combined with **K-Means** and **DBSCAN** clustering to identify "
            "distinct customer segments. Use the insights to drive targeted marketing "
            "campaigns, optimize retention, and maximize lifetime value."
        )
        st.divider()

        with st.expander("📖 Methodology", expanded=True):
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown("""
                **1. RFM Feature Engineering**
                - **Recency**: Days since last purchase
                - **Frequency**: Count of unique transactions
                - **Monetary**: Total spend (£)
                - Each metric scored 1–5 via quantile binning
                """)
            with col_m2:
                st.markdown("""
                **2. Rule-Based Segmentation**
                - 11 business segments mapped from R × FM scores
                - Segments like *Champions*, *At Risk*, *Lost*
                - Each segment has tailored marketing recommendations
                """)
            with col_m3:
                st.markdown("""
                **3. ML Clustering**
                - Log-transform + StandardScaler preprocessing
                - K-Means: Elbow method + Silhouette analysis
                - DBSCAN: Grid-search for density-based grouping
                - Cluster comparison & profiling
                """)

        st.divider()
        # Quick segment distribution
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_segment_treemap(rfm), use_container_width=True)
        with c2:
            st.plotly_chart(plot_revenue_by_segment(rfm), use_container_width=True)

    # ── Tab 2: RFM Analysis ──────────────────────────────────────────────
    with tab2:
        st.subheader("RFM Distributions & Segment Analysis")

        st.plotly_chart(plot_rfm_distributions(rfm), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_rfm_heatmap(rfm), use_container_width=True)
        with c2:
            st.plotly_chart(plot_segment_bar(rfm), use_container_width=True)

        st.plotly_chart(plot_snake_plot(rfm), use_container_width=True)

        st.plotly_chart(
            plot_rfm_3d_scatter(rfm, color_col="Segment"),
            use_container_width=True,
        )

    # ── Tab 3: Clustering ────────────────────────────────────────────────
    with tab3:
        st.subheader(f"K-Means Clustering (Optimal k = {optimal_k})")

        # Convert elbow results dict → DataFrame for visualization functions
        elbow_df = pd.DataFrame({
            "k": elbow["k_range"],
            "inertia": elbow["inertias"],
            "silhouette": elbow["silhouette_scores"],
        })

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_elbow(elbow_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_silhouette_comparison(elbow_df), use_container_width=True)

        # Add cluster labels to RFM for visualization
        rfm_clustered = rfm.copy()
        rfm_clustered["KMeans_Cluster"] = kmeans_result["labels"]
        rfm_clustered["KMeans_Cluster"] = rfm_clustered["KMeans_Cluster"].astype(str)

        st.plotly_chart(
            plot_rfm_3d_scatter(rfm_clustered, color_col="KMeans_Cluster"),
            use_container_width=True,
        )

        # Cluster profiles radar
        st.plotly_chart(plot_cluster_radar(kmeans_profiles), use_container_width=True)

        # Comparison table
        st.subheader("K-Means vs DBSCAN Comparison")
        st.dataframe(
            comparison_df.style.format({
                "Silhouette Score": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.subheader("K-Means Cluster Profiles")
        st.dataframe(kmeans_profiles.round(2), use_container_width=True)

    # ── Tab 4: Customer Lookup ───────────────────────────────────────────
    with tab4:
        st.subheader("🔍 Customer Lookup")
        st.markdown("Enter a Customer ID to view their RFM profile and segment.")

        cust_id_input = st.text_input(
            "Customer ID",
            placeholder="e.g., 12347",
        )

        if cust_id_input:
            try:
                cust_id_val = int(float(cust_id_input))
                cust_row = rfm[rfm["Customer ID"] == cust_id_val]

                if not cust_row.empty:
                    row = cust_row.iloc[0]
                    st.success(f"✅ Found Customer **{cust_id_val}**")

                    # Segment badge
                    seg = row["Segment"]
                    seg_color = SEGMENT_CONFIG.get(seg, {}).get("color", "#6C63FF")
                    seg_icon = SEGMENT_CONFIG.get(seg, {}).get("icon", "👤")
                    st.markdown(
                        f'<span class="segment-badge" style="background:{seg_color}; color:white;">'
                        f'{seg_icon} {seg}</span>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Metrics
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Recency", f"{row['Recency']:.0f} days")
                    m2.metric("Frequency", f"{row['Frequency']:.0f} orders")
                    m3.metric("Monetary", f"${row['Monetary']:,.2f}")
                    m4.metric("R Score", int(row["R_Score"]))
                    m5.metric("FM Score", int(row.get("FM_Score", 0)))

                    # Transaction history
                    st.markdown("---")
                    st.markdown("**Recent Transactions**")
                    cust_history = df_clean[
                        df_clean["Customer ID"] == cust_id_val
                    ].sort_values("InvoiceDate", ascending=False).head(10)

                    if not cust_history.empty:
                        display_cols = [c for c in ["Invoice", "InvoiceDate", "StockCode", "Description", "Quantity", "Price", "TotalPrice"] if c in cust_history.columns]
                        st.dataframe(cust_history[display_cols], use_container_width=True)
                    else:
                        st.info("No transaction history available in the cleaned dataset.")
                else:
                    st.warning(f"Customer ID **{cust_id_val}** not found in the dataset.")
            except ValueError:
                st.error("Please enter a valid numeric Customer ID.")

    # ── Tab 5: Segment Profiles ──────────────────────────────────────────
    with tab5:
        st.subheader("📋 Segment Profiles & Marketing Actions")

        summary_df = get_segment_summary(rfm)
        st.dataframe(
            summary_df.style.format({
                "avg_recency": "{:.1f}",
                "avg_frequency": "{:.1f}",
                "avg_monetary": "${:,.2f}",
                "pct": "{:.1f}%",
            }),
            use_container_width=True,
        )

        st.divider()
        st.subheader("🎯 Recommended Marketing Actions")

        actions = get_marketing_actions(rfm)
        for action in actions:
            seg_name = action["segment_name"]
            seg_cfg = SEGMENT_CONFIG.get(seg_name, {})
            color = seg_cfg.get("color", "#6C63FF")
            icon = seg_cfg.get("icon", "📌")
            rec = action["recommended_action"]
            count = action["customer_count"]

            st.markdown(f"""
            <div class="action-card" style="border-left: 4px solid {color};">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <div>
                        <strong style="color: {color}; font-size: 1.05rem;">{seg_name}</strong>
                        <span style="color: #718096; font-size: 0.85rem; margin-left: 8px;">
                            ({count:,} customers)
                        </span>
                        <p style="margin: 4px 0 0 0; color: #CBD5E0;">{rec}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Download button
        csv_data = rfm.to_csv(index=False)
        st.download_button(
            label="📥 Download Segmented Customer Data (CSV)",
            data=csv_data,
            file_name="segmented_customers.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
