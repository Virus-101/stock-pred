"""
LSTM Stock Price Prediction Model
==================================
A complete machine learning pipeline for stock price prediction using
Long Short-Term Memory (LSTM) neural networks.

Features:
- Data fetching and preprocessing
- Feature engineering with technical indicators
- LSTM model with configurable architecture
- Training with early stopping
- Comprehensive visualization
- Model evaluation metrics

Author: Created for Hicham Amini
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StockDataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def generate_sample_data(self, ticker="SAMPLE", days=1000):
        """
        Generate realistic sample stock data for demonstration.
        In production, replace with yfinance or other data source.
        """
        print(f"📊 Generating sample stock data for {ticker}...")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # Generate realistic price movement using geometric Brownian motion
        initial_price = 100
        mu = 0.0005  # Daily drift
        sigma = 0.02  # Daily volatility
        
        returns = np.random.normal(mu, sigma, days)
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        # Add some trend and seasonality
        trend = np.linspace(0, 50, days)
        seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, days))
        price_series = price_series + trend + seasonality
        
        # Generate OHLCV data
        high = price_series * (1 + np.abs(np.random.normal(0, 0.01, days)))
        low = price_series * (1 - np.abs(np.random.normal(0, 0.01, days)))
        open_price = low + (high - low) * np.random.random(days)
        volume = np.random.randint(1000000, 10000000, days)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price_series,
            'Volume': volume.astype(float)
        })
        
        df.set_index('Date', inplace=True)
        print(f"✅ Generated {len(df)} days of stock data")
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators as features."""
        print("🔧 Adding technical indicators...")
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Price Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Daily returns
        df['Returns'] = df['Close'].pct_change()
        df['Returns_Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        print(f"✅ Added {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} technical indicators")
        return df
    
    def prepare_sequences(self, df, target_col='Close', sequence_length=60, train_split=0.8):
        """Prepare sequences for LSTM training."""
        print(f"📦 Preparing sequences with lookback period of {sequence_length} days...")
        
        # Select features
        feature_columns = [col for col in df.columns if col != target_col]
        
        # Scale features
        features = df[feature_columns].values
        target = df[[target_col]].values
        
        features_scaled = self.scaler.fit_transform(features)
        target_scaled = self.price_scaler.fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Store dates for plotting
        self.train_dates = df.index[sequence_length:split_idx + sequence_length]
        self.test_dates = df.index[split_idx + sequence_length:]
        
        print(f"✅ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"   Sequence shape: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale."""
        return self.price_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


class LSTMStockPredictor:
    """LSTM model for stock price prediction."""
    
    def __init__(self, sequence_length, n_features, lstm_units=[128, 64], dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the LSTM model architecture."""
        print("🏗️ Building LSTM model...")
        
        self.model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(
                LSTM(self.lstm_units[0], return_sequences=True, 
                     input_shape=(self.sequence_length, self.n_features)),
                name='bidirectional_lstm_1'
            ),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second LSTM layer
            LSTM(self.lstm_units[1], return_sequences=True, name='lstm_2'),
            Dropout(self.dropout_rate, name='dropout_2'),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False, name='lstm_3'),
            Dropout(self.dropout_rate, name='dropout_3'),
            
            # Dense layers
            Dense(32, activation='relu', name='dense_1'),
            Dense(16, activation='relu', name='dense_2'),
            Dense(1, name='output')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Model built successfully!")
        self.model.summary()
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the LSTM model."""
        print(f"\n🚀 Training model for up to {epochs} epochs...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics


class StockVisualizer:
    """Visualization utilities for stock prediction analysis."""
    
    def __init__(self, figsize=(14, 8)):
        self.figsize = figsize
        
    def plot_stock_data(self, df, title="Stock Price History"):
        """Plot stock price with moving averages."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='#2E86AB')
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', linewidth=1, alpha=0.8, color='#F39C12')
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1, alpha=0.8, color='#E74C3C')
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1, color='gray', label='Bollinger Bands')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        ax2 = axes[1]
        colors = ['#27AE60' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#E74C3C' 
                  for i in range(len(df))]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=1)
        ax2.set_ylabel('Volume', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, history):
        """Plot training and validation loss."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#3498DB')
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#E74C3C')
        ax1.set_title('Model Loss During Training', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2 = axes[1]
        ax2.plot(history.history['mae'], label='Training MAE', linewidth=2, color='#3498DB')
        ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#E74C3C')
        ax2.set_title('Mean Absolute Error During Training', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('MAE', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, dates, y_true, y_pred, title="Stock Price Prediction"):
        """Plot actual vs predicted prices."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, y_true, label='Actual Price', linewidth=2, color='#2E86AB')
        ax.plot(dates, y_pred, label='Predicted Price', linewidth=2, color='#E74C3C', linestyle='--')
        
        ax.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_prediction_comparison(self, train_dates, train_true, train_pred,
                                   test_dates, test_true, test_pred):
        """Plot comprehensive prediction comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Full timeline
        ax1 = axes[0, 0]
        ax1.plot(train_dates, train_true, label='Training (Actual)', color='#2E86AB', alpha=0.7)
        ax1.plot(train_dates, train_pred, label='Training (Predicted)', color='#3498DB', linestyle='--', alpha=0.7)
        ax1.plot(test_dates, test_true, label='Test (Actual)', color='#27AE60', linewidth=2)
        ax1.plot(test_dates, test_pred, label='Test (Predicted)', color='#E74C3C', linestyle='--', linewidth=2)
        ax1.axvline(x=test_dates[0], color='gray', linestyle=':', linewidth=2, label='Train/Test Split')
        ax1.set_title('Complete Prediction Timeline', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Test period zoom
        ax2 = axes[0, 1]
        ax2.plot(test_dates, test_true, label='Actual', color='#27AE60', linewidth=2)
        ax2.plot(test_dates, test_pred, label='Predicted', color='#E74C3C', linewidth=2, linestyle='--')
        ax2.fill_between(test_dates, test_true, test_pred, alpha=0.2, color='orange')
        ax2.set_title('Test Period Predictions (Zoomed)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot
        ax3 = axes[1, 0]
        ax3.scatter(test_true, test_pred, alpha=0.5, color='#9B59B6', s=30)
        min_val = min(test_true.min(), test_pred.min())
        max_val = max(test_true.max(), test_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax3.set_title('Actual vs Predicted (Scatter)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Actual Price ($)', fontsize=11)
        ax3.set_ylabel('Predicted Price ($)', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error distribution
        ax4 = axes[1, 1]
        errors = test_true - test_pred
        ax4.hist(errors, bins=50, color='#3498DB', alpha=0.7, edgecolor='white')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(x=errors.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
        ax4.set_title('Prediction Error Distribution', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Prediction Error ($)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_summary(self, metrics):
        """Create a visual summary of model metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Normalize for visualization (different scales)
        colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
        
        bars = ax.barh(metric_names, metric_values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            ax.text(bar.get_width() + 0.01 * max(metric_values), bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig


def main():
    """Main execution pipeline."""
    print("=" * 60)
    print("🔮 LSTM Stock Price Prediction Model")
    print("=" * 60)
    print()
    
    # Configuration
    SEQUENCE_LENGTH = 60  # Look back 60 days
    TRAIN_SPLIT = 0.8
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Initialize components
    processor = StockDataProcessor()
    visualizer = StockVisualizer()
    
    # Step 1: Load and process data
    print("\n" + "=" * 40)
    print("STEP 1: Data Loading & Processing")
    print("=" * 40)
    
    df = processor.generate_sample_data(ticker="DEMO", days=1000)
    df = processor.add_technical_indicators(df)
    
    print(f"\n📈 Dataset shape: {df.shape}")
    print(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"\nFeatures: {list(df.columns)}")
    
    # Visualize raw data
    fig1 = visualizer.plot_stock_data(df, title="Sample Stock Price with Technical Indicators")
    fig1.savefig('./stock_data_analysis.png', dpi=150, bbox_inches='tight')
    print("\n💾 Saved: stock_data_analysis.png")
    
    # Step 2: Prepare sequences
    print("\n" + "=" * 40)
    print("STEP 2: Preparing Data Sequences")
    print("=" * 40)
    
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_sequences(
        df, target_col='Close', sequence_length=SEQUENCE_LENGTH, train_split=TRAIN_SPLIT
    )
    
    # Step 3: Build and train model
    print("\n" + "=" * 40)
    print("STEP 3: Building & Training LSTM Model")
    print("=" * 40)
    
    predictor = LSTMStockPredictor(
        sequence_length=SEQUENCE_LENGTH,
        n_features=X_train.shape[2],
        lstm_units=[128, 64],
        dropout_rate=0.2
    )
    
    predictor.build_model()
    
    history = predictor.train(
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Plot training history
    fig2 = visualizer.plot_training_history(history)
    fig2.savefig('./training_history.png', dpi=150, bbox_inches='tight')
    print("\n💾 Saved: training_history.png")
    
    # Step 4: Make predictions
    print("\n" + "=" * 40)
    print("STEP 4: Making Predictions")
    print("=" * 40)
    
    # Predictions
    train_pred_scaled = predictor.predict(X_train)
    test_pred_scaled = predictor.predict(X_test)
    
    # Inverse transform
    train_pred = processor.inverse_transform_predictions(train_pred_scaled)
    test_pred = processor.inverse_transform_predictions(test_pred_scaled)
    y_train_orig = processor.inverse_transform_predictions(y_train)
    y_test_orig = processor.inverse_transform_predictions(y_test)
    
    # Step 5: Evaluate model
    print("\n" + "=" * 40)
    print("STEP 5: Model Evaluation")
    print("=" * 40)
    
    train_metrics = predictor.evaluate(y_train_orig, train_pred)
    test_metrics = predictor.evaluate(y_test_orig, test_pred)
    
    print("\n📊 Training Set Metrics:")
    for name, value in train_metrics.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n📊 Test Set Metrics:")
    for name, value in test_metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Step 6: Visualizations
    print("\n" + "=" * 40)
    print("STEP 6: Generating Visualizations")
    print("=" * 40)
    
    # Comprehensive prediction comparison
    fig3 = visualizer.plot_prediction_comparison(
        processor.train_dates, y_train_orig, train_pred,
        processor.test_dates, y_test_orig, test_pred
    )
    fig3.savefig('./prediction_comparison.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: prediction_comparison.png")
    
    # Test predictions zoom
    fig4 = visualizer.plot_predictions(
        processor.test_dates, y_test_orig, test_pred,
        title="Test Set: Actual vs Predicted Stock Prices"
    )
    fig4.savefig('./test_predictions.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: test_predictions.png")
    
    # Metrics summary
    fig5 = visualizer.plot_metrics_summary(test_metrics)
    fig5.savefig('./metrics_summary.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: metrics_summary.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ LSTM Stock Prediction Model Complete!")
    print("=" * 60)
    print(f"""
📁 Generated Files:
   • stock_data_analysis.png    - Raw data visualization
   • training_history.png       - Training loss curves
   • prediction_comparison.png  - Full prediction analysis
   • test_predictions.png       - Test set predictions
   • metrics_summary.png        - Performance metrics

📈 Model Performance Summary:
   • R² Score: {test_metrics['R2']:.4f} (1.0 = perfect)
   • RMSE: ${test_metrics['RMSE']:.2f}
   • MAPE: {test_metrics['MAPE']:.2f}%

🔧 Model Configuration:
   • Sequence Length: {SEQUENCE_LENGTH} days
   • LSTM Architecture: Bidirectional(128) → LSTM(64) → LSTM(32) → Dense
   • Features Used: {len(feature_names)} technical indicators
   • Training Split: {TRAIN_SPLIT*100:.0f}% train / {(1-TRAIN_SPLIT)*100:.0f}% test
""")
    
    plt.close('all')
    return predictor, processor, test_metrics


if __name__ == "__main__":
    predictor, processor, metrics = main()
