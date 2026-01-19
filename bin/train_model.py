#!/usr/bin/env python3

"""
Train ML model for soil moisture prediction.

This script trains an LSTM neural network to predict future soil moisture
levels based on historical data, enabling smarter irrigation decisions.

Features used:
- Historical soil moisture values (sequence)
- Soil temperature
- Day of year (seasonality)
- Hour of day (daily patterns)

Usage:
    ./train_model.py --input soil_data.csv --output model.pt \
        --sequence-length 24 --epochs 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoilMoistureDataset(Dataset):
    """Dataset for soil moisture time series prediction."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


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
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        # Predict future moisture values
        predictions = self.fc(last_hidden)
        return predictions


def load_and_preprocess_data(input_file: str) -> pd.DataFrame:
    """Load and preprocess soil moisture data."""
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    logger.info(f"Loaded {len(df)} records")
    return df


def create_sequences(df: pd.DataFrame, sequence_length: int,
                     forecast_horizon: int, feature_cols: List[str],
                     target_col: str = 'soil_moisture') -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Create sequences for LSTM training.

    Args:
        df: Input dataframe
        sequence_length: Number of past timesteps to use
        forecast_horizon: Number of future timesteps to predict
        feature_cols: Columns to use as features
        target_col: Column to predict

    Returns:
        Tuple of (sequences, targets, feature_scaler, target_scaler)
    """
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    logger.info(f"Using features: {available_cols}")

    # Scale features
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features = df[available_cols].values
    targets = df[[target_col]].values

    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets)

    # Create sequences
    sequences = []
    target_sequences = []

    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        seq = features_scaled[i:i + sequence_length]
        target = targets_scaled[i + sequence_length:i + sequence_length + forecast_horizon].flatten()

        sequences.append(seq)
        target_sequences.append(target)

    return (np.array(sequences), np.array(target_sequences),
            feature_scaler, target_scaler)


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, epochs: int,
                learning_rate: float, device: torch.device) -> Dict:
    """Train the LSTM model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    model.to(device)
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {avg_val_loss:.6f}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return history


def save_model(model: nn.Module, feature_scaler: MinMaxScaler,
               target_scaler: MinMaxScaler, config: Dict,
               output_path: str, metadata_path: str):
    """Save model and associated metadata."""
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config,
    }, output_path)

    # Save scalers and metadata as JSON-compatible format
    metadata = {
        'model_type': 'SoilMoistureLSTM',
        'created_at': datetime.now().isoformat(),
        'config': config,
        'feature_scaler': {
            'min': feature_scaler.data_min_.tolist(),
            'max': feature_scaler.data_max_.tolist(),
            'scale': feature_scaler.scale_.tolist(),
        },
        'target_scaler': {
            'min': target_scaler.data_min_.tolist(),
            'max': target_scaler.data_max_.tolist(),
            'scale': target_scaler.scale_.tolist(),
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {output_path}")
    logger.info(f"Metadata saved to {metadata_path}")


def run_training(args) -> Dict:
    """Run the complete training pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    df = load_and_preprocess_data(args.input)

    # Define features
    feature_cols = [
        'soil_moisture',
        'soil_temp_surface',
        'soil_temp_10cm',
        'hour',
        'day_of_year',
    ]

    # Create sequences
    try:
        sequences, targets, feature_scaler, target_scaler = create_sequences(
            df,
            sequence_length=args.sequence_length,
            forecast_horizon=args.forecast_horizon,
            feature_cols=feature_cols
        )
    except ValueError as e:
        logger.error(f"Failed to create sequences: {e}")
        return {'error': str(e)}

    min_sequences = args.min_sequences if hasattr(args, 'min_sequences') else 10
    if len(sequences) < min_sequences:
        error_msg = (f"Insufficient data for ML training. "
                     f"Got {len(sequences)} sequences, need at least {min_sequences}. "
                     f"Original records: {len(df)}. "
                     f"Try using a longer date range to fetch more historical data.")
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Created {len(sequences)} sequences")

    # Split data
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]

    # Create datasets
    train_dataset = SoilMoistureDataset(train_sequences, train_targets)
    val_dataset = SoilMoistureDataset(val_sequences, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model configuration
    input_size = sequences.shape[2]
    config = {
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'sequence_length': args.sequence_length,
        'forecast_horizon': args.forecast_horizon,
        'feature_cols': [c for c in feature_cols if c in df.columns],
    }

    # Create model
    model = SoilMoistureLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        forecast_horizon=args.forecast_horizon
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    # Save model
    save_model(model, feature_scaler, target_scaler, config,
               args.output, args.metadata)

    results = {
        'status': 'success',
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'best_val_loss': min(history['val_loss']),
        'epochs_trained': len(history['train_loss']),
        'training_samples': len(train_sequences),
        'validation_samples': len(val_sequences),
    }

    logger.info(f"Training complete: {results}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for soil moisture prediction"
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file with soil moisture data')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output model file (.pt)')
    parser.add_argument('--metadata', '-m', type=str, required=True,
                        help='Output metadata file (.json)')

    # Model architecture
    parser.add_argument('--sequence-length', type=int, default=24,
                        help='Number of past timesteps to use')
    parser.add_argument('--forecast-horizon', type=int, default=24,
                        help='Number of future timesteps to predict')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--min-sequences', type=int, default=10,
                        help='Minimum number of sequences required for training (fails if not met)')

    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
