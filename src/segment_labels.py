"""
Module for human-readable segment labels and marketing recommendations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SEGMENT_CONFIG = {
    'Champions': {
        'description': 'Bought recently, buy often, and spend the most.',
        'recommendation': 'Reward them. They can be early adopters for new products. Will promote your brand.',
        'color': '#28a745',
        'icon': '🏆'
    },
    'Loyal Customers': {
        'description': 'Spend good money with us often. Responsive to promotions.',
        'recommendation': 'Upsell higher value products. Ask for reviews.',
        'color': '#17a2b8',
        'icon': '🤝'
    },
    'Potential Loyalists': {
        'description': 'Recent customers, but spent a good amount and bought more than once.',
        'recommendation': 'Offer membership / loyalty program. Keep them engaged.',
        'color': '#20c997',
        'icon': '🌟'
    },
    'Recent Customers': {
        'description': 'Bought most recently, but not often.',
        'recommendation': 'Provide onboarding support, give them early success, offer a welcome discount.',
        'color': '#007bff',
        'icon': '👋'
    },
    'Promising': {
        'description': 'Recent shoppers, but haven’t spent much.',
        'recommendation': 'Create brand awareness, offer free trials.',
        'color': '#6610f2',
        'icon': '🌱'
    },
    'Need Attention': {
        'description': 'Above average recency, frequency and monetary values. May not have bought very recently though.',
        'recommendation': 'Make limited time offers, recommend based on past purchases.',
        'color': '#fd7e14',
        'icon': '👀'
    },
    'About to Sleep': {
        'description': 'Below average recency, frequency and monetary values. Will lose them if not reactivated.',
        'recommendation': 'Share valuable resources, recommend popular products/renewals at discount.',
        'color': '#ffc107',
        'icon': '😴'
    },
    'At Risk': {
        'description': 'Spent big money and purchased often. But long time ago. Need to bring them back!',
        'recommendation': 'Send personalized emails to reconnect, offer renewals, provide helpful resources.',
        'color': '#dc3545',
        'icon': '⚠️'
    },
    'Can\'t Lose': {
        'description': 'Made biggest purchases, and often. But haven’t returned for a long time.',
        'recommendation': 'Win them back via renewals or newer products, don’t lose them to competition.',
        'color': '#c82333',
        'icon': '🚨'
    },
    'Hibernating': {
        'description': 'Last purchase was long back, low spenders and low number of orders.',
        'recommendation': 'Offer other relevant products and special discounts.',
        'color': '#6c757d',
        'icon': '💤'
    },
    'Lost': {
        'description': 'Lowest recency, frequency and monetary scores.',
        'recommendation': 'Revive interest with reach out campaign, otherwise ignore.',
        'color': '#343a40',
        'icon': '💔'
    }
}

def label_clusters_by_rfm(rfm_df: pd.DataFrame, cluster_col: str = 'Cluster') -> pd.DataFrame:
    """
    Adds a 'Cluster_Label' column to rfm_df by analyzing the mean R, F, M scores per cluster
    and assigning a composite ranking.
    """
    df = rfm_df.copy()
    
    # Calculate mean scores per cluster
    cluster_means = df.groupby(cluster_col)[['R_Score', 'F_Score', 'M_Score']].mean()
    
    # Create a composite score
    cluster_means['composite_score'] = cluster_means['R_Score'] + cluster_means['F_Score'] + cluster_means['M_Score']
    
    # Rank clusters based on composite score
    ranked_clusters = cluster_means.sort_values(by='composite_score', ascending=False).index.tolist()
    
    # Map cluster to an arbitrary label based on rank (best to worst)
    labels_ordered = ['Champions', 'Loyal Customers', 'Need Attention', 'At Risk', 'Hibernating', 'Lost']
    
    label_map = {}
    for i, cluster_id in enumerate(ranked_clusters):
        # Assign a label based on the relative position
        idx = min(i * len(labels_ordered) // len(ranked_clusters), len(labels_ordered) - 1)
        label_map[cluster_id] = labels_ordered[idx]
        
    df['Cluster_Label'] = df[cluster_col].map(label_map)
    return df

def get_segment_summary(rfm_df: pd.DataFrame, segment_col: str = 'Segment') -> pd.DataFrame:
    """
    Returns a summary DataFrame with count, pct, and averages per segment.
    """
    agg_df = rfm_df.groupby(segment_col).agg(
        avg_recency=('Recency', 'mean'),
        avg_frequency=('Frequency', 'mean'),
        avg_monetary=('Monetary', 'mean'),
        count=('Recency', 'size'),
    )
    
    agg_df['pct'] = (agg_df['count'] / agg_df['count'].sum()) * 100
    
    # Add recommendation
    agg_df['recommendation'] = agg_df.index.map(
        lambda x: SEGMENT_CONFIG.get(x, {}).get('recommendation', 'No recommendation')
    )
    
    return agg_df.reset_index()

def get_marketing_actions(rfm_df: pd.DataFrame, segment_col: str = 'Segment') -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with segment name, customer count, and recommended action.
    """
    counts = rfm_df[segment_col].value_counts().to_dict()
    actions = []
    
    for segment, count in counts.items():
        actions.append({
            'segment_name': segment,
            'customer_count': count,
            'recommended_action': SEGMENT_CONFIG.get(segment, {}).get('recommendation', 'No action available.')
        })
        
    return actions
