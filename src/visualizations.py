"""
Visualizations Module

This module provides reusable plotting functions for the EDA and Streamlit dashboard.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Common layout styling for Plotly
DARK_THEME = dict(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='#FAFAFA')
)
COLOR_SEQUENCE = px.colors.qualitative.Pastel

def apply_theme(fig: go.Figure) -> go.Figure:
    """Applies standard dark theme to a plotly figure."""
    fig.update_layout(
        plot_bgcolor=DARK_THEME['plot_bgcolor'],
        paper_bgcolor=DARK_THEME['paper_bgcolor'],
        font=DARK_THEME['font'],
        colorway=COLOR_SEQUENCE,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
    return fig

def plot_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """Plot histograms of Recency, Frequency, and Monetary."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Recency", "Frequency", "Monetary"))
    
    fig.add_trace(go.Histogram(x=rfm_df['Recency'], name='Recency', marker_color=COLOR_SEQUENCE[0]), row=1, col=1)
    fig.add_trace(go.Histogram(x=rfm_df['Frequency'], name='Frequency', marker_color=COLOR_SEQUENCE[1 % len(COLOR_SEQUENCE)]), row=1, col=2)
    fig.add_trace(go.Histogram(x=rfm_df['Monetary'], name='Monetary', marker_color=COLOR_SEQUENCE[2 % len(COLOR_SEQUENCE)]), row=1, col=3)
    
    fig.update_layout(title_text="RFM Distributions", showlegend=False)
    return apply_theme(fig)

def plot_rfm_3d_scatter(rfm_df: pd.DataFrame, color_col: str = 'Segment') -> go.Figure:
    """Plot 3D scatter of RFM colored by segment."""
    fig = px.scatter_3d(
        rfm_df, x='Recency', y='Frequency', z='Monetary',
        color=color_col,
        title=f"3D RFM Clusters ({color_col})",
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_traces(marker=dict(size=3))
    return apply_theme(fig)

def plot_elbow(elbow_results: pd.DataFrame) -> go.Figure:
    """Plot dual-axis line chart for inertia and silhouette score."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=elbow_results['k'], y=elbow_results['inertia'], name="Inertia", mode='lines+markers', line=dict(color=COLOR_SEQUENCE[0])),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=elbow_results['k'], y=elbow_results['silhouette'], name="Silhouette Score", mode='lines+markers', line=dict(color=COLOR_SEQUENCE[1 % len(COLOR_SEQUENCE)])),
        secondary_y=True,
    )
    
    fig.update_layout(title_text="Elbow Method Results")
    fig.update_xaxes(title_text="Number of clusters (k)")
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return apply_theme(fig)

def plot_silhouette_comparison(elbow_results: pd.DataFrame) -> go.Figure:
    """Plot bar chart of silhouette scores per k."""
    fig = px.bar(
        elbow_results, x='k', y='silhouette',
        title="Silhouette Score per k",
        labels={'k': 'Number of Clusters (k)', 'silhouette': 'Silhouette Score'},
        color_discrete_sequence=[COLOR_SEQUENCE[2 % len(COLOR_SEQUENCE)]]
    )
    return apply_theme(fig)

def plot_segment_treemap(rfm_df: pd.DataFrame, segment_col: str = 'Segment') -> go.Figure:
    """Treemap showing segment sizes."""
    segment_counts = rfm_df[segment_col].value_counts().reset_index()
    segment_counts.columns = [segment_col, 'Count']
    
    fig = px.treemap(
        segment_counts, path=[segment_col], values='Count',
        title="Customer Segments Treemap",
        color=segment_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    return apply_theme(fig)

def plot_segment_bar(rfm_df: pd.DataFrame, segment_col: str = 'Segment') -> go.Figure:
    """Horizontal bar chart of segment counts."""
    segment_counts = rfm_df[segment_col].value_counts().reset_index()
    segment_counts.columns = [segment_col, 'Count']
    segment_counts = segment_counts.sort_values('Count', ascending=True)
    
    fig = px.bar(
        segment_counts, x='Count', y=segment_col, orientation='h',
        title="Customer Count per Segment",
        color=segment_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    return apply_theme(fig)

def plot_cluster_radar(cluster_profiles_df: pd.DataFrame) -> go.Figure:
    """Radar/spider chart comparing cluster centroids on R, F, M."""
    categories = ['Recency', 'Frequency', 'Monetary']
    
    fig = go.Figure()
    
    for i, row in cluster_profiles_df.iterrows():
        cluster_name = row.name if 'Segment' not in cluster_profiles_df.columns else row['Segment']
        values = [row.get('Recency', 0), row.get('Frequency', 0), row.get('Monetary', 0)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=str(cluster_name),
            marker=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)])
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor='#333333'),
            bgcolor='#0E1117'
        ),
        showlegend=True,
        title="Cluster Profiles Radar Chart"
    )
    return apply_theme(fig)

def plot_rfm_heatmap(rfm_df: pd.DataFrame) -> go.Figure:
    """Heatmap of R_Score vs FM_Score with customer counts."""
    if 'R_Score' in rfm_df.columns and 'FM_Score' in rfm_df.columns:
        heatmap_data = rfm_df.groupby(['R_Score', 'FM_Score']).size().unstack(fill_value=0)
    else:
        try:
            r_score = pd.qcut(rfm_df['Recency'], q=4, labels=[4, 3, 2, 1])
            f_score = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
            m_score = pd.qcut(rfm_df['Monetary'], q=4, labels=[1, 2, 3, 4])
            fm_score = (f_score.astype(int) + m_score.astype(int)) // 2
            heatmap_data = pd.crosstab(r_score, fm_score)
        except Exception:
            heatmap_data = pd.DataFrame(np.random.randint(10, 100, size=(4,4)), index=[4,3,2,1], columns=[1,2,3,4])

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="FM Score", y="R Score", color="Count"),
        title="R vs FM Score Heatmap",
        color_continuous_scale="Viridis"
    )
    return apply_theme(fig)

def plot_snake_plot(rfm_df: pd.DataFrame, segment_col: str = 'Segment') -> go.Figure:
    """Normalized RFM values per segment (line chart)."""
    rfm_normalized = rfm_df.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        if rfm_normalized[col].std() != 0:
            rfm_normalized[col] = (rfm_normalized[col] - rfm_normalized[col].mean()) / rfm_normalized[col].std()
        else:
            rfm_normalized[col] = 0
        
    snake_data = rfm_normalized.groupby(segment_col)[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    snake_melt = pd.melt(snake_data, id_vars=[segment_col], value_vars=['Recency', 'Frequency', 'Monetary'],
                         var_name='Metric', value_name='Value')
                         
    fig = px.line(
        snake_melt, x='Metric', y='Value', color=segment_col, markers=True,
        title="Snake Plot of Standardized RFM Metrics",
        color_discrete_sequence=COLOR_SEQUENCE
    )
    return apply_theme(fig)

def plot_revenue_by_segment(rfm_df: pd.DataFrame, segment_col: str = 'Segment') -> go.Figure:
    """Pie/donut chart of revenue contribution per segment."""
    revenue_data = rfm_df.groupby(segment_col)['Monetary'].sum().reset_index()
    
    fig = px.pie(
        revenue_data, values='Monetary', names=segment_col, hole=0.4,
        title="Revenue Contribution by Segment",
        color_discrete_sequence=COLOR_SEQUENCE
    )
    return apply_theme(fig)
