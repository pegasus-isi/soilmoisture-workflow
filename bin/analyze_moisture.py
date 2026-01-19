#!/usr/bin/env python3

"""
Analyze soil moisture data and calculate irrigation metrics.

This script processes soil moisture data to:
1. Calculate moisture statistics
2. Identify moisture trends
3. Compute soil water deficit
4. Classify moisture levels by crop requirements

Usage:
    ./analyze_moisture.py --input soil_data.csv --output analysis.json \
        --crop-type tomato --soil-type loam
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Crop water requirements (soil moisture thresholds in m3/m3)
CROP_THRESHOLDS = {
    'tomato': {'wilting': 0.10, 'stress': 0.20, 'optimal_low': 0.25, 'optimal_high': 0.35, 'saturated': 0.45},
    'corn': {'wilting': 0.08, 'stress': 0.15, 'optimal_low': 0.20, 'optimal_high': 0.30, 'saturated': 0.40},
    'wheat': {'wilting': 0.08, 'stress': 0.12, 'optimal_low': 0.18, 'optimal_high': 0.28, 'saturated': 0.38},
    'lettuce': {'wilting': 0.12, 'stress': 0.22, 'optimal_low': 0.28, 'optimal_high': 0.38, 'saturated': 0.45},
    'potato': {'wilting': 0.10, 'stress': 0.18, 'optimal_low': 0.25, 'optimal_high': 0.35, 'saturated': 0.42},
    'grape': {'wilting': 0.06, 'stress': 0.12, 'optimal_low': 0.18, 'optimal_high': 0.28, 'saturated': 0.35},
    'alfalfa': {'wilting': 0.08, 'stress': 0.15, 'optimal_low': 0.22, 'optimal_high': 0.32, 'saturated': 0.40},
    'cotton': {'wilting': 0.08, 'stress': 0.14, 'optimal_low': 0.20, 'optimal_high': 0.30, 'saturated': 0.38},
    'default': {'wilting': 0.10, 'stress': 0.18, 'optimal_low': 0.25, 'optimal_high': 0.35, 'saturated': 0.42},
}

# Soil water holding capacity (m3/m3)
SOIL_CAPACITY = {
    'sand': {'field_capacity': 0.15, 'wilting_point': 0.05, 'saturation': 0.40},
    'sandy_loam': {'field_capacity': 0.22, 'wilting_point': 0.08, 'saturation': 0.45},
    'loam': {'field_capacity': 0.30, 'wilting_point': 0.12, 'saturation': 0.48},
    'clay_loam': {'field_capacity': 0.35, 'wilting_point': 0.15, 'saturation': 0.50},
    'clay': {'field_capacity': 0.40, 'wilting_point': 0.20, 'saturation': 0.52},
    'default': {'field_capacity': 0.30, 'wilting_point': 0.12, 'saturation': 0.48},
}


def load_data(input_file: str) -> pd.DataFrame:
    """Load soil moisture data."""
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Loaded {len(df)} records")
    return df


def classify_moisture(moisture: float, crop_type: str) -> str:
    """Classify moisture level based on crop requirements."""
    thresholds = CROP_THRESHOLDS.get(crop_type, CROP_THRESHOLDS['default'])

    if moisture <= thresholds['wilting']:
        return 'critical'
    elif moisture <= thresholds['stress']:
        return 'stressed'
    elif moisture <= thresholds['optimal_low']:
        return 'low'
    elif moisture <= thresholds['optimal_high']:
        return 'optimal'
    elif moisture <= thresholds['saturated']:
        return 'high'
    else:
        return 'saturated'


def calculate_water_deficit(moisture: float, soil_type: str, crop_type: str) -> Dict:
    """
    Calculate soil water deficit.

    Returns:
        Dict with deficit metrics and irrigation recommendation
    """
    soil = SOIL_CAPACITY.get(soil_type, SOIL_CAPACITY['default'])
    crop = CROP_THRESHOLDS.get(crop_type, CROP_THRESHOLDS['default'])

    # Available water capacity
    awc = soil['field_capacity'] - soil['wilting_point']

    # Current available water
    if moisture > soil['wilting_point']:
        available_water = moisture - soil['wilting_point']
    else:
        available_water = 0

    # Deficit from field capacity
    deficit_from_fc = max(0, soil['field_capacity'] - moisture)

    # Deficit from optimal range
    target = (crop['optimal_low'] + crop['optimal_high']) / 2
    deficit_from_optimal = max(0, target - moisture)

    # Relative saturation (0-1)
    if awc > 0:
        relative_saturation = available_water / awc
    else:
        relative_saturation = 0

    return {
        'soil_moisture': moisture,
        'field_capacity': soil['field_capacity'],
        'wilting_point': soil['wilting_point'],
        'available_water': round(available_water, 4),
        'available_water_capacity': round(awc, 4),
        'deficit_from_field_capacity': round(deficit_from_fc, 4),
        'deficit_from_optimal': round(deficit_from_optimal, 4),
        'relative_saturation': round(relative_saturation, 4),
        'saturation_percent': round(relative_saturation * 100, 1),
    }


def calculate_trend(df: pd.DataFrame, window_hours: int = 24) -> Dict:
    """Calculate moisture trend over time window."""
    if 'timestamp' not in df.columns or 'soil_moisture' not in df.columns:
        return {'trend': 'unknown', 'change_rate': 0}

    df_sorted = df.sort_values('timestamp')
    df_recent = df_sorted.tail(window_hours * 2)  # Assume ~2 readings per hour max

    if len(df_recent) < 2:
        return {'trend': 'insufficient_data', 'change_rate': 0}

    moisture_values = df_recent['soil_moisture'].dropna()
    if len(moisture_values) < 2:
        return {'trend': 'insufficient_data', 'change_rate': 0}

    # Calculate slope
    x = np.arange(len(moisture_values))
    slope, _ = np.polyfit(x, moisture_values.values, 1)

    # Classify trend
    if slope > 0.001:
        trend = 'increasing'
    elif slope < -0.001:
        trend = 'decreasing'
    else:
        trend = 'stable'

    # Calculate daily change rate
    time_span = (df_recent['timestamp'].max() - df_recent['timestamp'].min()).total_seconds() / 3600
    if time_span > 0:
        hourly_change = slope / time_span
        daily_change = hourly_change * 24
    else:
        daily_change = 0

    return {
        'trend': trend,
        'hourly_change_rate': round(slope, 6),
        'daily_change_rate': round(daily_change, 6),
        'observation_hours': round(time_span, 1),
    }


def analyze_polygon(df: pd.DataFrame, polygon_id: str, crop_type: str,
                    soil_type: str) -> Dict:
    """Analyze moisture data for a single polygon."""
    polygon_df = df[df['polygon_id'] == polygon_id].copy()

    if polygon_df.empty:
        return {'error': f'No data for polygon {polygon_id}'}

    moisture_col = 'soil_moisture'
    if moisture_col not in polygon_df.columns:
        return {'error': 'No soil_moisture column in data'}

    moisture_values = polygon_df[moisture_col].dropna()

    if moisture_values.empty:
        return {'error': 'No valid moisture values'}

    # Get latest reading
    latest_idx = polygon_df['timestamp'].idxmax() if 'timestamp' in polygon_df.columns else moisture_values.index[-1]
    latest_moisture = polygon_df.loc[latest_idx, moisture_col]

    # Basic statistics
    stats = {
        'polygon_id': polygon_id,
        'polygon_name': polygon_df['polygon_name'].iloc[0] if 'polygon_name' in polygon_df.columns else '',
        'crop_type': crop_type,
        'soil_type': soil_type,
        'analysis_timestamp': datetime.now().isoformat(),
        'data_points': len(moisture_values),
        'time_range': {
            'start': polygon_df['timestamp'].min().isoformat() if 'timestamp' in polygon_df.columns else None,
            'end': polygon_df['timestamp'].max().isoformat() if 'timestamp' in polygon_df.columns else None,
        },
        'moisture_stats': {
            'current': round(latest_moisture, 4),
            'mean': round(moisture_values.mean(), 4),
            'std': round(moisture_values.std(), 4),
            'min': round(moisture_values.min(), 4),
            'max': round(moisture_values.max(), 4),
            'median': round(moisture_values.median(), 4),
        },
        'moisture_classification': classify_moisture(latest_moisture, crop_type),
        'water_deficit': calculate_water_deficit(latest_moisture, soil_type, crop_type),
        'trend': calculate_trend(polygon_df),
    }

    # Add temperature stats if available
    for temp_col in ['soil_temp_surface', 'soil_temp_10cm']:
        if temp_col in polygon_df.columns:
            temp_values = polygon_df[temp_col].dropna()
            if not temp_values.empty:
                stats[f'{temp_col}_stats'] = {
                    'current': round(temp_values.iloc[-1], 1),
                    'mean': round(temp_values.mean(), 1),
                    'min': round(temp_values.min(), 1),
                    'max': round(temp_values.max(), 1),
                }

    # Classification distribution
    classifications = polygon_df[moisture_col].apply(
        lambda x: classify_moisture(x, crop_type) if pd.notna(x) else 'unknown'
    )
    stats['classification_distribution'] = classifications.value_counts().to_dict()

    return stats


def run_analysis(input_file: str, output_file: str, crop_type: str,
                 soil_type: str) -> Dict:
    """Run complete analysis on soil moisture data."""
    df = load_data(input_file)

    if df.empty:
        logger.error("No data to analyze")
        return {}

    # Get unique polygons
    polygon_ids = df['polygon_id'].unique() if 'polygon_id' in df.columns else ['default']

    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'input_file': input_file,
            'crop_type': crop_type,
            'soil_type': soil_type,
            'total_records': len(df),
            'polygons_analyzed': len(polygon_ids),
        },
        'crop_thresholds': CROP_THRESHOLDS.get(crop_type, CROP_THRESHOLDS['default']),
        'soil_properties': SOIL_CAPACITY.get(soil_type, SOIL_CAPACITY['default']),
        'polygon_analyses': [],
    }

    for polygon_id in polygon_ids:
        logger.info(f"Analyzing polygon: {polygon_id}")
        analysis = analyze_polygon(df, polygon_id, crop_type, soil_type)
        results['polygon_analyses'].append(analysis)

    # Summary across all polygons
    all_moisture = df['soil_moisture'].dropna()
    if not all_moisture.empty:
        results['summary'] = {
            'overall_mean_moisture': round(all_moisture.mean(), 4),
            'polygons_needing_water': sum(
                1 for a in results['polygon_analyses']
                if a.get('moisture_classification') in ['critical', 'stressed', 'low']
            ),
            'polygons_optimal': sum(
                1 for a in results['polygon_analyses']
                if a.get('moisture_classification') == 'optimal'
            ),
            'polygons_saturated': sum(
                1 for a in results['polygon_analyses']
                if a.get('moisture_classification') in ['high', 'saturated']
            ),
        }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Analysis saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze soil moisture data for irrigation decisions"
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file with soil moisture data')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSON file for analysis results')
    parser.add_argument('--crop-type', type=str, default='default',
                        choices=list(CROP_THRESHOLDS.keys()),
                        help='Crop type for threshold selection')
    parser.add_argument('--soil-type', type=str, default='loam',
                        choices=list(SOIL_CAPACITY.keys()),
                        help='Soil type for water capacity calculation')

    args = parser.parse_args()

    results = run_analysis(args.input, args.output, args.crop_type, args.soil_type)

    if results:
        summary = results.get('summary', {})
        logger.info(f"Analysis complete: {summary.get('polygons_needing_water', 0)} polygon(s) need water")


if __name__ == "__main__":
    main()
