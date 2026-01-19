#!/usr/bin/env python3

"""
Predict irrigation needs using ML and rule-based methods.

This script combines:
1. LSTM-based soil moisture forecasting
2. Rule-based thresholds for crop requirements
3. Weather forecast integration

Usage:
    ./predict_irrigation.py --analysis analysis.json --weather weather.json \
        --model model.pt --model-metadata model_metadata.json \
        --output irrigation_prediction.json
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ML imports (optional - graceful fallback if not available)
try:
    import torch
    import torch.nn as nn
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("PyTorch not available - using rule-based predictions only")


# Evapotranspiration coefficients (simplified Penman-Monteith factors)
ET_COEFFICIENTS = {
    'tomato': 1.15,
    'corn': 1.20,
    'wheat': 1.15,
    'lettuce': 1.00,
    'potato': 1.15,
    'grape': 0.85,
    'alfalfa': 1.20,
    'cotton': 1.20,
    'default': 1.10,
}


# ML Model Definition (must match train_model.py)
if ML_AVAILABLE:
    class SoilMoistureLSTM(nn.Module):
        """LSTM model for soil moisture prediction."""

        def __init__(self, input_size: int, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2,
                     forecast_horizon: int = 24):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.forecast_horizon = forecast_horizon

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, forecast_horizon)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            predictions = self.fc(last_hidden)
            return predictions


class MLPredictor:
    """Handles ML-based soil moisture prediction."""

    def __init__(self, model_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        self.model = None
        self.metadata = None
        self.device = None
        self.is_loaded = False

        if model_path and metadata_path and ML_AVAILABLE:
            self._load_model(model_path, metadata_path)

    def _load_model(self, model_path: str, metadata_path: str):
        """Load trained model and metadata."""
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Check for insufficient data marker
            if self.metadata.get('error') == 'insufficient_data':
                logger.warning("Model was trained with insufficient data - using rule-based fallback")
                return

            config = self.metadata.get('config', {})

            # Load model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.model = SoilMoistureLSTM(
                input_size=config.get('input_size', 5),
                hidden_size=config.get('hidden_size', 64),
                num_layers=config.get('num_layers', 2),
                forecast_horizon=config.get('forecast_horizon', 24)
            )

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"ML model loaded successfully from {model_path}")

        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self.is_loaded = False

    def predict_moisture(self, recent_data: pd.DataFrame) -> Optional[Dict]:
        """
        Predict future soil moisture using the trained model.

        Args:
            recent_data: DataFrame with recent soil moisture readings

        Returns:
            Dict with predictions or None if prediction fails
        """
        if not self.is_loaded or self.model is None:
            return None

        try:
            config = self.metadata.get('config', {})
            feature_cols = config.get('feature_cols', ['soil_moisture'])
            sequence_length = config.get('sequence_length', 24)
            forecast_horizon = config.get('forecast_horizon', 24)

            # Prepare features
            available_cols = [c for c in feature_cols if c in recent_data.columns]
            if not available_cols:
                return None

            # Get last sequence_length rows
            if len(recent_data) < sequence_length:
                # Pad with repeated values
                padding_needed = sequence_length - len(recent_data)
                padding = pd.concat([recent_data.iloc[[0]]] * padding_needed)
                recent_data = pd.concat([padding, recent_data]).reset_index(drop=True)

            sequence_data = recent_data[available_cols].tail(sequence_length).values

            # Scale using saved scaler parameters
            scaler_info = self.metadata.get('feature_scaler', {})
            if scaler_info:
                data_min = np.array(scaler_info['min'])
                data_max = np.array(scaler_info['max'])
                data_range = data_max - data_min
                data_range[data_range == 0] = 1  # Avoid division by zero
                sequence_scaled = (sequence_data - data_min) / data_range
            else:
                sequence_scaled = sequence_data

            # Predict
            with torch.no_grad():
                input_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
                predictions_scaled = self.model(input_tensor).cpu().numpy()[0]

            # Inverse scale predictions
            target_scaler = self.metadata.get('target_scaler', {})
            if target_scaler:
                t_min = target_scaler['min'][0]
                t_max = target_scaler['max'][0]
                predictions = predictions_scaled * (t_max - t_min) + t_min
            else:
                predictions = predictions_scaled

            # Ensure predictions are in valid range
            predictions = np.clip(predictions, 0.0, 0.55)

            return {
                'forecast_hours': list(range(1, forecast_horizon + 1)),
                'predicted_moisture': predictions.tolist(),
                'mean_predicted': float(np.mean(predictions)),
                'min_predicted': float(np.min(predictions)),
                'max_predicted': float(np.max(predictions)),
                'trend': 'decreasing' if predictions[-1] < predictions[0] else 'increasing',
                'hours_until_critical': self._hours_until_threshold(predictions, 0.10),
                'hours_until_stress': self._hours_until_threshold(predictions, 0.18),
            }

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def _hours_until_threshold(self, predictions: np.ndarray, threshold: float) -> Optional[int]:
        """Find hours until moisture drops below threshold."""
        below_threshold = np.where(predictions < threshold)[0]
        if len(below_threshold) > 0:
            return int(below_threshold[0]) + 1
        return None


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def estimate_daily_et(temp_c: float, humidity: float, wind_speed: float,
                      crop_type: str) -> float:
    """
    Estimate daily evapotranspiration (mm/day) using simplified Hargreaves method.

    Args:
        temp_c: Temperature in Celsius
        humidity: Relative humidity (%)
        wind_speed: Wind speed (m/s)
        crop_type: Crop for Kc coefficient

    Returns:
        Estimated ET in mm/day
    """
    # Reference ET (simplified)
    # Using modified Hargreaves: ET0 = 0.0023 * (Tmean + 17.8) * Ra^0.5 * TD^0.5
    # Simplified for daily estimate
    if temp_c < 0:
        base_et = 0.5
    elif temp_c < 10:
        base_et = 1.0 + (temp_c / 10)
    elif temp_c < 25:
        base_et = 2.0 + (temp_c - 10) * 0.15
    else:
        base_et = 4.0 + (temp_c - 25) * 0.2

    # Humidity adjustment
    humidity_factor = 1.0 + (50 - humidity) * 0.01 if humidity < 50 else 1.0 - (humidity - 50) * 0.005

    # Wind adjustment
    wind_factor = 1.0 + wind_speed * 0.05

    # Crop coefficient
    kc = ET_COEFFICIENTS.get(crop_type, ET_COEFFICIENTS['default'])

    et = base_et * humidity_factor * wind_factor * kc
    return max(0.5, min(et, 12.0))  # Clamp to reasonable range


def predict_moisture_tomorrow(current_moisture: float, et_rate: float,
                              rain_forecast: float, soil_type: str) -> float:
    """
    Predict tomorrow's soil moisture based on ET and rain.

    Args:
        current_moisture: Current soil moisture (m3/m3)
        et_rate: Evapotranspiration rate (mm/day)
        rain_forecast: Forecasted rain (mm)
        soil_type: Soil type for infiltration estimate

    Returns:
        Predicted soil moisture (m3/m3)
    """
    # Convert ET to moisture loss (rough estimate: 1mm = 0.001 m3/m3 for 1m root zone)
    et_loss = et_rate * 0.001

    # Rain infiltration efficiency by soil type
    infiltration = {
        'sand': 0.90,
        'sandy_loam': 0.85,
        'loam': 0.75,
        'clay_loam': 0.60,
        'clay': 0.50,
        'default': 0.70,
    }
    rain_eff = infiltration.get(soil_type, 0.70)

    # Rain contribution to soil moisture
    rain_gain = rain_forecast * rain_eff * 0.001

    predicted = current_moisture - et_loss + rain_gain

    # Clamp to valid range
    return max(0.0, min(predicted, 0.55))


def calculate_irrigation_need(analysis: Dict, weather_forecast: Optional[Dict] = None,
                              ml_prediction: Optional[Dict] = None) -> Dict:
    """
    Calculate irrigation needs based on analysis, weather, and ML predictions.

    Args:
        analysis: Soil moisture analysis results
        weather_forecast: Optional weather forecast data
        ml_prediction: Optional ML-based moisture predictions

    Returns:
        Irrigation recommendation with urgency and amount
    """
    moisture_class = analysis.get('moisture_classification', 'unknown')
    water_deficit = analysis.get('water_deficit', {})
    trend = analysis.get('trend', {})
    crop_type = analysis.get('crop_type', 'default')
    soil_type = analysis.get('soil_type', 'loam')

    current_moisture = water_deficit.get('soil_moisture', 0.25)
    deficit_optimal = water_deficit.get('deficit_from_optimal', 0)
    deficit_fc = water_deficit.get('deficit_from_field_capacity', 0)
    saturation_pct = water_deficit.get('saturation_percent', 50)

    # Base decision on moisture classification
    decision = {
        'polygon_id': analysis.get('polygon_id'),
        'polygon_name': analysis.get('polygon_name'),
        'current_moisture': current_moisture,
        'moisture_classification': moisture_class,
        'recommendation_timestamp': datetime.now().isoformat(),
    }

    # Calculate urgency score (0-100)
    if moisture_class == 'critical':
        urgency = 100
        action = 'irrigate_immediately'
        reason = 'Soil moisture at critical level - crop damage likely'
    elif moisture_class == 'stressed':
        urgency = 80
        action = 'irrigate_soon'
        reason = 'Plants experiencing water stress'
    elif moisture_class == 'low':
        urgency = 50
        action = 'schedule_irrigation'
        reason = 'Soil moisture below optimal range'
    elif moisture_class == 'optimal':
        urgency = 20
        action = 'monitor'
        reason = 'Soil moisture in optimal range'
    elif moisture_class == 'high':
        urgency = 10
        action = 'no_irrigation'
        reason = 'Soil moisture above optimal - avoid overwatering'
    else:  # saturated
        urgency = 0
        action = 'no_irrigation'
        reason = 'Soil saturated - risk of root damage'

    # Adjust for trend
    if trend.get('trend') == 'decreasing' and urgency < 80:
        urgency = min(100, urgency + 15)
        reason += '; moisture declining'
    elif trend.get('trend') == 'increasing' and urgency > 20:
        urgency = max(0, urgency - 10)
        reason += '; moisture increasing'

    # Weather forecast adjustment
    rain_expected = 0
    if weather_forecast:
        # Sum rain forecast for next 24-48 hours
        forecasts = weather_forecast.get('forecasts', [])
        for fc in forecasts[:16]:  # Next 48 hours (3-hour intervals)
            rain_expected += fc.get('rain', {}).get('3h', 0)

        if rain_expected > 10:  # Significant rain expected
            urgency = max(0, urgency - 30)
            action = 'wait_for_rain'
            reason += f'; {rain_expected:.1f}mm rain forecasted'
        elif rain_expected > 5:
            urgency = max(0, urgency - 15)
            reason += f'; light rain ({rain_expected:.1f}mm) expected'

    # ML prediction adjustments
    ml_insights = {}
    if ml_prediction:
        hours_until_critical = ml_prediction.get('hours_until_critical')
        hours_until_stress = ml_prediction.get('hours_until_stress')
        predicted_trend = ml_prediction.get('trend', 'unknown')
        min_predicted = ml_prediction.get('min_predicted', current_moisture)

        ml_insights = {
            'ml_prediction_available': True,
            'predicted_min_moisture': round(min_predicted, 4),
            'predicted_trend': predicted_trend,
            'hours_until_critical': hours_until_critical,
            'hours_until_stress': hours_until_stress,
        }

        # Adjust urgency based on ML predictions
        if hours_until_critical is not None:
            if hours_until_critical <= 6:
                urgency = max(urgency, 95)
                action = 'irrigate_immediately'
                reason += f'; ML predicts critical moisture in {hours_until_critical}h'
            elif hours_until_critical <= 24:
                urgency = max(urgency, 75)
                action = 'irrigate_soon' if action not in ['irrigate_immediately'] else action
                reason += f'; ML predicts critical in {hours_until_critical}h'

        elif hours_until_stress is not None:
            if hours_until_stress <= 12:
                urgency = max(urgency, 60)
                if action not in ['irrigate_immediately', 'irrigate_soon']:
                    action = 'schedule_irrigation'
                reason += f'; ML predicts stress in {hours_until_stress}h'

        # Reduce urgency if ML predicts improvement
        if predicted_trend == 'increasing' and urgency > 30:
            urgency = max(20, urgency - 15)
            reason += '; ML predicts moisture recovery'
    else:
        ml_insights = {'ml_prediction_available': False}

    # Calculate recommended irrigation amount (mm)
    if action in ['irrigate_immediately', 'irrigate_soon', 'schedule_irrigation']:
        # Target: bring soil to field capacity
        # Approximate: 1mm irrigation = 0.001 m3/m3 soil moisture for 1m depth
        target_increase = deficit_optimal + 0.02  # Slightly above optimal
        irrigation_mm = target_increase * 1000 * 0.3  # Assume 0.3m effective root zone

        # Adjust for rain
        irrigation_mm = max(0, irrigation_mm - rain_expected * 0.7)
    else:
        irrigation_mm = 0

    decision.update({
        'urgency_score': round(urgency),
        'action': action,
        'reason': reason,
        'irrigation_amount_mm': round(irrigation_mm, 1),
        'rain_forecasted_mm': round(rain_expected, 1),
        'deficit_from_optimal': round(deficit_optimal, 4),
        'saturation_percent': round(saturation_pct, 1),
        'trend': trend.get('trend', 'unknown'),
        'ml_insights': ml_insights,
    })

    # Add estimated timing
    if urgency >= 80:
        decision['recommended_timing'] = 'within 6 hours'
    elif urgency >= 50:
        decision['recommended_timing'] = 'within 24 hours'
    elif urgency >= 30:
        decision['recommended_timing'] = 'within 48 hours'
    else:
        decision['recommended_timing'] = 'not needed'

    return decision


def run_prediction(analysis_file: str, weather_file: Optional[str],
                   output_file: str, model_file: str,
                   metadata_file: str,
                   soil_data_file: str) -> Dict:
    """Run irrigation prediction with ML model (required)."""
    logger.info("Loading analysis data...")
    analysis_data = load_json(analysis_file)

    weather_data = None
    if weather_file:
        try:
            weather_data = load_json(weather_file)
            logger.info("Weather forecast loaded")
        except Exception as e:
            logger.warning(f"Could not load weather data: {e}")

    # Initialize ML predictor (required)
    logger.info("Loading ML model...")
    ml_predictor = MLPredictor(model_file, metadata_file)
    if not ml_predictor.is_loaded:
        raise RuntimeError(f"Failed to load ML model from {model_file}. "
                           "ML is required for predictions. "
                           "Ensure the model was trained successfully.")

    logger.info("ML predictor initialized successfully")

    # Load soil data for ML predictions (required)
    try:
        soil_data = pd.read_csv(soil_data_file)
        if 'timestamp' in soil_data.columns:
            soil_data['timestamp'] = pd.to_datetime(soil_data['timestamp'])
            soil_data = soil_data.sort_values('timestamp')
        # Add time features for ML
        if 'timestamp' in soil_data.columns:
            soil_data['hour'] = soil_data['timestamp'].dt.hour
            soil_data['day_of_year'] = soil_data['timestamp'].dt.dayofyear
        logger.info(f"Loaded {len(soil_data)} soil data records for ML prediction")
    except Exception as e:
        raise RuntimeError(f"Failed to load soil data from {soil_data_file}: {e}")

    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'analysis_file': analysis_file,
            'weather_file': weather_file,
            'model_file': model_file,
            'ml_enabled': True,
        },
        'predictions': [],
        'summary': {
            'total_polygons': 0,
            'need_immediate_irrigation': 0,
            'need_scheduled_irrigation': 0,
            'optimal_no_action': 0,
            'total_irrigation_mm': 0,
            'ml_predictions_made': 0,
        }
    }

    polygon_analyses = analysis_data.get('polygon_analyses', [])

    for analysis in polygon_analyses:
        if 'error' in analysis:
            continue

        # Get ML prediction for this polygon (required)
        polygon_id = analysis.get('polygon_id')
        if 'polygon_id' in soil_data.columns:
            polygon_data = soil_data[soil_data['polygon_id'] == polygon_id]
        else:
            polygon_data = soil_data

        ml_prediction = None
        if not polygon_data.empty:
            ml_prediction = ml_predictor.predict_moisture(polygon_data)
            if ml_prediction:
                results['summary']['ml_predictions_made'] += 1
            else:
                logger.warning(f"ML prediction failed for polygon {polygon_id}")

        prediction = calculate_irrigation_need(analysis, weather_data, ml_prediction)
        results['predictions'].append(prediction)

        # Update summary
        results['summary']['total_polygons'] += 1
        action = prediction.get('action', '')

        if action == 'irrigate_immediately':
            results['summary']['need_immediate_irrigation'] += 1
        elif action in ['irrigate_soon', 'schedule_irrigation']:
            results['summary']['need_scheduled_irrigation'] += 1
        else:
            results['summary']['optimal_no_action'] += 1

        results['summary']['total_irrigation_mm'] += prediction.get('irrigation_amount_mm', 0)

    # Overall recommendation
    if results['summary']['need_immediate_irrigation'] > 0:
        results['summary']['overall_action'] = 'URGENT: Some fields need immediate irrigation'
    elif results['summary']['need_scheduled_irrigation'] > 0:
        results['summary']['overall_action'] = 'SCHEDULE: Some fields need irrigation soon'
    else:
        results['summary']['overall_action'] = 'OK: All fields have adequate moisture'

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Summary: {results['summary']['overall_action']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict irrigation needs based on soil moisture analysis"
    )

    parser.add_argument('--analysis', '-a', type=str, required=True,
                        help='Input analysis JSON file')
    parser.add_argument('--weather', '-w', type=str,
                        help='Optional weather forecast JSON file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output prediction JSON file')

    # ML model arguments (required)
    parser.add_argument('--model', type=str, required=True,
                        help='Trained ML model file (.pt)')
    parser.add_argument('--model-metadata', type=str, required=True,
                        help='ML model metadata file (.json)')
    parser.add_argument('--soil-data', type=str, required=True,
                        help='Raw soil data CSV for ML predictions')

    args = parser.parse_args()

    run_prediction(
        args.analysis,
        args.weather,
        args.output,
        model_file=args.model,
        metadata_file=args.model_metadata,
        soil_data_file=args.soil_data
    )


if __name__ == "__main__":
    main()
