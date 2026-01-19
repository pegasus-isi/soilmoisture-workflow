#!/usr/bin/env python3

"""
Visualize soil moisture data and irrigation recommendations.

Creates multi-panel visualizations showing:
1. Soil moisture time series with thresholds
2. Irrigation urgency gauge
3. Moisture trend analysis
4. Forecast and recommendations

Usage:
    ./visualize_moisture.py --data soil_data.csv --analysis analysis.json \
        --prediction prediction.json --output moisture_viz.png
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Color scheme for moisture levels
MOISTURE_COLORS = {
    'critical': '#d62728',      # Red
    'stressed': '#ff7f0e',      # Orange
    'low': '#ffbb78',           # Light orange
    'optimal': '#2ca02c',       # Green
    'high': '#1f77b4',          # Blue
    'saturated': '#9467bd',     # Purple
}


def load_data(data_file: str) -> pd.DataFrame:
    """Load soil moisture CSV data."""
    df = pd.read_csv(data_file)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_moisture_timeseries(ax, df: pd.DataFrame, thresholds: Dict,
                               polygon_name: str = ''):
    """Create soil moisture time series plot with threshold bands."""
    if 'timestamp' not in df.columns or 'soil_moisture' not in df.columns:
        ax.text(0.5, 0.5, 'No time series data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    df_sorted = df.sort_values('timestamp')

    # Plot moisture values
    ax.plot(df_sorted['timestamp'], df_sorted['soil_moisture'],
            'b-', linewidth=1.5, label='Soil Moisture', zorder=5)
    ax.scatter(df_sorted['timestamp'], df_sorted['soil_moisture'],
               c='blue', s=20, alpha=0.6, zorder=6)

    # Add threshold bands
    ylim = ax.get_ylim()
    y_max = max(0.5, df_sorted['soil_moisture'].max() * 1.2)

    # Critical zone
    ax.axhspan(0, thresholds.get('wilting', 0.10),
               alpha=0.2, color=MOISTURE_COLORS['critical'], label='Critical')

    # Stressed zone
    ax.axhspan(thresholds.get('wilting', 0.10), thresholds.get('stress', 0.18),
               alpha=0.2, color=MOISTURE_COLORS['stressed'], label='Stressed')

    # Low zone
    ax.axhspan(thresholds.get('stress', 0.18), thresholds.get('optimal_low', 0.25),
               alpha=0.2, color=MOISTURE_COLORS['low'], label='Low')

    # Optimal zone
    ax.axhspan(thresholds.get('optimal_low', 0.25), thresholds.get('optimal_high', 0.35),
               alpha=0.3, color=MOISTURE_COLORS['optimal'], label='Optimal')

    # High zone
    ax.axhspan(thresholds.get('optimal_high', 0.35), thresholds.get('saturated', 0.45),
               alpha=0.2, color=MOISTURE_COLORS['high'], label='High')

    # Saturated zone
    ax.axhspan(thresholds.get('saturated', 0.45), y_max,
               alpha=0.2, color=MOISTURE_COLORS['saturated'], label='Saturated')

    ax.set_ylim(0, y_max)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Soil Moisture (m³/m³)', fontsize=10)
    ax.set_title(f'Soil Moisture Time Series - {polygon_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def create_urgency_gauge(ax, urgency: float, action: str):
    """Create an urgency gauge visualization."""
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # Background arc segments
    segments = [
        (0, 20, MOISTURE_COLORS['optimal']),
        (20, 50, MOISTURE_COLORS['low']),
        (50, 80, MOISTURE_COLORS['stressed']),
        (80, 100, MOISTURE_COLORS['critical']),
    ]

    for start, end, color in segments:
        start_angle = np.pi * (1 - start/100)
        end_angle = np.pi * (1 - end/100)
        theta_seg = np.linspace(end_angle, start_angle, 50)
        x = np.cos(theta_seg)
        y = np.sin(theta_seg)
        ax.fill_between(x, 0, y, alpha=0.3, color=color)

    # Needle
    needle_angle = np.pi * (1 - urgency/100)
    ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
             head_width=0.08, head_length=0.05, fc='black', ec='black', linewidth=2)

    # Center circle
    circle = plt.Circle((0, 0), 0.1, color='black', zorder=10)
    ax.add_patch(circle)

    # Labels
    ax.text(0, -0.2, f'{urgency:.0f}', fontsize=24, fontweight='bold',
            ha='center', va='top')
    ax.text(0, -0.4, action.replace('_', ' ').upper(), fontsize=10,
            ha='center', va='top', color='gray')

    # Scale labels
    for val, label in [(0, '0'), (25, '25'), (50, '50'), (75, '75'), (100, '100')]:
        angle = np.pi * (1 - val/100)
        ax.text(1.15 * np.cos(angle), 1.15 * np.sin(angle), label,
                fontsize=8, ha='center', va='center')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.6, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Irrigation Urgency', fontsize=12, fontweight='bold')


def create_water_status(ax, water_deficit: Dict, crop_type: str):
    """Create water status bar chart."""
    labels = ['Current', 'Optimal', 'Field Cap.']
    values = [
        water_deficit.get('soil_moisture', 0) * 100,
        (water_deficit.get('soil_moisture', 0) + water_deficit.get('deficit_from_optimal', 0)) * 100,
        water_deficit.get('field_capacity', 0.30) * 100,
    ]

    colors = [MOISTURE_COLORS['optimal'] if values[0] >= values[1] * 0.9
              else MOISTURE_COLORS['stressed'], '#888888', '#888888']

    bars = ax.barh(labels, values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    # Add wilting point line
    wp = water_deficit.get('wilting_point', 0.12) * 100
    ax.axvline(wp, color='red', linestyle='--', linewidth=2, label='Wilting Point')

    ax.set_xlim(0, 60)
    ax.set_xlabel('Volumetric Water Content (%)', fontsize=10)
    ax.set_title(f'Water Status ({crop_type})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)


def create_recommendation_panel(ax, prediction: Dict):
    """Create text panel with recommendations."""
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Irrigation Recommendation', fontsize=14, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Recommendation details
    action = prediction.get('action', 'unknown').replace('_', ' ').title()
    timing = prediction.get('recommended_timing', 'N/A')
    amount = prediction.get('irrigation_amount_mm', 0)
    rain = prediction.get('rain_forecasted_mm', 0)
    reason = prediction.get('reason', '')

    # Color based on urgency
    urgency = prediction.get('urgency_score', 50)
    if urgency >= 80:
        box_color = MOISTURE_COLORS['critical']
    elif urgency >= 50:
        box_color = MOISTURE_COLORS['stressed']
    elif urgency >= 30:
        box_color = MOISTURE_COLORS['low']
    else:
        box_color = MOISTURE_COLORS['optimal']

    # Action box
    box = mpatches.FancyBboxPatch((0.05, 0.55), 0.9, 0.3,
                                   boxstyle="round,pad=0.02",
                                   facecolor=box_color, alpha=0.3,
                                   transform=ax.transAxes)
    ax.add_patch(box)

    ax.text(0.5, 0.75, action, fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.62, f'Timing: {timing}', fontsize=11,
            ha='center', va='center', transform=ax.transAxes)

    # Details
    details = [
        f"Irrigation needed: {amount:.1f} mm" if amount > 0 else "No irrigation needed",
        f"Rain forecasted: {rain:.1f} mm" if rain > 0 else "No rain expected",
        "",
        f"Reason: {reason[:60]}..." if len(reason) > 60 else f"Reason: {reason}",
    ]

    y_pos = 0.45
    for detail in details:
        ax.text(0.5, y_pos, detail, fontsize=9,
                ha='center', va='center', transform=ax.transAxes)
        y_pos -= 0.08


def create_trend_indicator(ax, trend: Dict, moisture_stats: Dict):
    """Create trend indicator panel."""
    trend_dir = trend.get('trend', 'unknown')
    daily_change = trend.get('daily_change_rate', 0)

    # Arrow based on trend
    if trend_dir == 'increasing':
        arrow = '↑'
        color = MOISTURE_COLORS['optimal']
    elif trend_dir == 'decreasing':
        arrow = '↓'
        color = MOISTURE_COLORS['stressed']
    else:
        arrow = '→'
        color = 'gray'

    ax.text(0.5, 0.7, arrow, fontsize=60, ha='center', va='center',
            color=color, transform=ax.transAxes)
    ax.text(0.5, 0.35, trend_dir.upper(), fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.2, f'{daily_change*100:+.2f}%/day', fontsize=10,
            ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.set_title('Moisture Trend', fontsize=12, fontweight='bold')
    ax.axis('off')


def create_visualization(data_file: str, analysis_file: str,
                         prediction_file: str, output_file: str):
    """Create complete visualization."""
    logger.info("Creating visualization...")

    # Load data
    df = load_data(data_file)
    analysis = load_json(analysis_file)
    prediction_data = load_json(prediction_file)

    # Get first polygon analysis and prediction
    polygon_analysis = analysis.get('polygon_analyses', [{}])[0]
    prediction = prediction_data.get('predictions', [{}])[0]
    thresholds = analysis.get('crop_thresholds', {})

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Layout: 3 rows, 3 columns
    # Row 1: Time series (spans 2 cols) + Gauge
    # Row 2: Water status + Trend + Recommendation
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)  # Time series
    ax2 = plt.subplot2grid((2, 3), (0, 2))              # Gauge
    ax3 = plt.subplot2grid((2, 3), (1, 0))              # Water status
    ax4 = plt.subplot2grid((2, 3), (1, 1))              # Trend
    ax5 = plt.subplot2grid((2, 3), (1, 2))              # Recommendation

    # Create panels
    polygon_name = polygon_analysis.get('polygon_name', 'Field')
    create_moisture_timeseries(ax1, df, thresholds, polygon_name)

    urgency = prediction.get('urgency_score', 50)
    action = prediction.get('action', 'monitor')
    create_urgency_gauge(ax2, urgency, action)

    water_deficit = polygon_analysis.get('water_deficit', {})
    crop_type = polygon_analysis.get('crop_type', 'default')
    create_water_status(ax3, water_deficit, crop_type)

    trend = polygon_analysis.get('trend', {})
    moisture_stats = polygon_analysis.get('moisture_stats', {})
    create_trend_indicator(ax4, trend, moisture_stats)

    create_recommendation_panel(ax5, prediction)

    # Title
    fig.suptitle(f'Soil Moisture Analysis - Should I Water?',
                 fontsize=16, fontweight='bold', y=0.98)

    # Timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=8, ha='right', va='bottom', color='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_file, dpi=150, facecolor='white', edgecolor='none')
    plt.close()

    logger.info(f"Visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize soil moisture data and irrigation recommendations"
    )

    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Input soil data CSV file')
    parser.add_argument('--analysis', '-a', type=str, required=True,
                        help='Analysis JSON file')
    parser.add_argument('--prediction', '-p', type=str, required=True,
                        help='Prediction JSON file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PNG file')

    args = parser.parse_args()

    create_visualization(args.data, args.analysis, args.prediction, args.output)


if __name__ == "__main__":
    main()
