"""
Cost Predictor Model
LSTM-based model for predicting future cloud costs based on historical usage patterns
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class CloudCostPredictor:
    """
    LSTM-based model for predicting future cloud costs
    """
    
    def __init__(
        self,
        lookback_window: int = 14,
        forecast_horizon: int = 7,
        model_path: Optional[str] = None
    ):
        """
        Initialize Cost Predictor
        
        Args:
            lookback_window: Number of past days to use for prediction
            forecast_horizon: Number of future days to predict
            model_path: Path to save/load model
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.history = None
        self.model_path = model_path or "results/models/saved_models"
        
        # Create model directory
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
    
    def build_model(self, n_features: int) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            n_features: Number of input features
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                128,
                activation='relu',
                return_sequences=True,
                input_shape=(self.lookback_window, n_features),
                name='lstm_1'
            ),
            layers.Dropout(0.2, name='dropout_1'),
            
            # Second LSTM layer
            layers.LSTM(
                64,
                activation='relu',
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Dense layers
            layers.Dense(32, activation='relu', name='dense_1'),
            layers.Dropout(0.1, name='dropout_3'),
            
            # Output layer (forecast_horizon predictions)
            layers.Dense(self.forecast_horizon, name='output')
        ])
        
        # Compile with custom metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=[
                'mae',
                'mape',
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        
        self.model = model
        return model
    
    def prepare_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Normalized data array
        
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape (n_samples, lookback_window, n_features)
            - y: Target sequences of shape (n_samples, forecast_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            # Input: lookback_window days of all features
            X.append(data[i:i+self.lookback_window])
            
            # Target: forecast_horizon days of cost only (first column)
            y.append(data[i+self.lookback_window:
                         i+self.lookback_window+self.forecast_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        data: pd.DataFrame,
        feature_columns: list = None,
        target_column: str = 'cost',
        test_size: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping: bool = True,
        verbose: int = 1
    ) -> Dict:
        """
        Train the cost prediction model
        
        Args:
            data: DataFrame with historical data
            feature_columns: List of feature column names to use
            target_column: Target column name (cost)
            test_size: Fraction of data for testing
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation during training
            early_stopping: Whether to use early stopping
            verbose: Verbosity level
        
        Returns:
            Dictionary with training metrics
        """
        # Default features if not specified
        if feature_columns is None:
            feature_columns = [
                'cost', 'cpu_usage', 'memory_usage',
                'network_traffic', 'storage_usage', 'request_count'
            ]
        
        # Ensure target is first column
        if target_column not in feature_columns:
            feature_columns = [target_column] + feature_columns
        else:
            feature_columns = [target_column] + [c for c in feature_columns if c != target_column]
        
        self.feature_names = feature_columns
        
        # Extract features
        features_data = data[feature_columns].values
        
        # Normalize data
        print("Normalizing data...")
        self.scaler = StandardScaler()
        data_normalized = self.scaler.fit_transform(features_data)
        
        # Prepare sequences
        print("Preparing sequences...")
        X, y = self.prepare_sequences(data_normalized)
        
        print(f"Sequences prepared: X shape={X.shape}, y shape={y.shape}")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Build model if not exists
        if self.model is None:
            print("Building model...")
            self.build_model(n_features=len(feature_columns))
            print(self.model.summary())
        
        # Callbacks
        callbacks = []
        
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        )
        
        # Train model
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Denormalize predictions and actuals
        y_pred_denorm = self._denormalize_predictions(y_pred)
        y_test_denorm = self._denormalize_predictions(y_test)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(y_pred_denorm - y_test_denorm))
        mape = np.mean(np.abs((y_test_denorm - y_pred_denorm) / y_test_denorm)) * 100
        rmse = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
        
        results = {
            'test_loss': test_metrics[0],
            'test_mae': mae,
            'test_mape': mape,
            'test_rmse': rmse,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(self.history.history['loss'])
        }
        
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Test MAE:  ${mae:.2f}")
        print(f"Test MAPE: {mape:.2f}%")
        print(f"Test RMSE: ${rmse:.2f}")
        print("="*60)
        
        return results
    
    def _denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale"""
        # Create array with predictions in first column, zeros for other features
        n_samples = predictions.shape[0]
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        n_features = len(self.feature_names)
        full_array = np.zeros((n_samples, n_features))
        full_array[:, 0] = predictions.mean(axis=1) if len(predictions.shape) > 1 else predictions.flatten()
        
        # Inverse transform
        denormalized = self.scaler.inverse_transform(full_array)
        
        return denormalized[:, 0]
    
    def predict(
        self,
        recent_data: pd.DataFrame,
        return_confidence: bool = False
    ) -> Dict:
        """
        Predict future costs
        
        Args:
            recent_data: DataFrame with recent data (at least lookback_window days)
            return_confidence: Whether to return confidence intervals
        
        Returns:
            Dictionary with predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features_data = recent_data[self.feature_names].values
        
        # Take last lookback_window days
        if len(features_data) < self.lookback_window:
            raise ValueError(
                f"Need at least {self.lookback_window} days of data, "
                f"got {len(features_data)}"
            )
        
        features_data = features_data[-self.lookback_window:]
        
        # Normalize
        data_normalized = self.scaler.transform(features_data)
        
        # Prepare input
        X = data_normalized.reshape(1, self.lookback_window, -1)
        
        # Predict
        predictions_normalized = self.model.predict(X, verbose=0)[0]
        
        # Denormalize
        predictions = self._denormalize_predictions(predictions_normalized)
        
        # Generate dates for predictions
        last_date = recent_data['timestamp'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_horizon,
            freq='D'
        )
        
        result = {
            'predictions': predictions.tolist(),
            'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'total_predicted_cost': float(predictions.sum()),
            'avg_daily_cost': float(predictions.mean()),
            'min_cost': float(predictions.min()),
            'max_cost': float(predictions.max())
        }
        
        if return_confidence:
            # Simple confidence interval (could be improved with MC dropout)
            std = predictions.std()
            result['confidence_lower'] = (predictions - 1.96 * std).tolist()
            result['confidence_upper'] = (predictions + 1.96 * std).tolist()
        
        return result
    
    def save(self, name: str = "cost_predictor"):
        """Save model and scaler"""
        # Save model
        model_file = f"{self.model_path}/{name}_model.h5"
        self.model.save(model_file)
        print(f"✓ Model saved: {model_file}")
        
        # Save scaler
        scaler_file = f"{self.model_path}/{name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        print(f"✓ Scaler saved: {scaler_file}")
        
        # Save configuration
        config = {
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'feature_names': self.feature_names
        }
        config_file = f"{self.model_path}/{name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved: {config_file}")
    
    def load(self, name: str = "cost_predictor"):
        """Load model and scaler"""
        # Load config
        config_file = f"{self.model_path}/{name}_config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.lookback_window = config['lookback_window']
        self.forecast_horizon = config['forecast_horizon']
        self.feature_names = config['feature_names']
        
        # Load model
        model_file = f"{self.model_path}/{name}_model.h5"
        self.model = keras.models.load_model(model_file)
        
        # Load scaler
        scaler_file = f"{self.model_path}/{name}_scaler.pkl"
        self.scaler = joblib.load(scaler_file)
        
        print(f"✓ Model loaded from {model_file}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE
        axes[1, 0].plot(self.history.history['mape'], label='Train MAPE')
        axes[1, 0].plot(self.history.history['val_mape'], label='Val MAPE')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RMSE
        axes[1, 1].plot(self.history.history['rmse'], label='Train RMSE')
        axes[1, 1].plot(self.history.history['val_rmse'], label='Val RMSE')
        axes[1, 1].set_title('Root Mean Squared Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Demo
    print("Loading data...")
    data = pd.read_csv('data/sample_data/daily_usage.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    print(f"Data loaded: {len(data)} days")
    
    # Initialize predictor
    predictor = CloudCostPredictor(
        lookback_window=14,
        forecast_horizon=7
    )
    
    # Train
    results = predictor.train(
        data=data,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    predictor.save("cost_predictor_v1")
    
    # Make prediction
    recent_data = data.tail(14)
    prediction = predictor.predict(recent_data, return_confidence=True)
    
    print("\n" + "="*60)
    print("PREDICTION FOR NEXT 7 DAYS")
    print("="*60)
    print(f"Total predicted cost: ${prediction['total_predicted_cost']:.2f}")
    print(f"Average daily cost: ${prediction['avg_daily_cost']:.2f}")
    print("\nDaily predictions:")
    for date, cost in zip(prediction['dates'], prediction['predictions']):
        print(f"  {date}: ${cost:.2f}")
    
    # Plot training history
    predictor.plot_training_history(
        save_path='results/training_history.png'
    )
