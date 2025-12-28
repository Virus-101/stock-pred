"""
LSTM Stock Price Prediction - Real Data Version
================================================
Uses yfinance to fetch real stock data from Yahoo Finance.

Usage:
    python stock_prediction_real.py --ticker AAPL --days 1000
    python stock_prediction_real.py --ticker MSFT --days 500 --epochs 50
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Install with: pip install yfinance")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-darkgrid')


class RealStockDataProcessor:
    """Fetches and processes real stock data."""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_dates = None
        self.test_dates = None
        
    def fetch_stock_data(self, ticker, period="5y"):
        """Fetch real stock data using yfinance."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        print(f"📥 Fetching {ticker} data from Yahoo Finance...")
        
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Clean up columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone info
        
        print(f"✅ Fetched {len(df)} trading days of data")
        print(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators."""
        print("🔧 Computing technical indicators...")
        
        # Price-based features
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] if 'EMA_12' in df.columns else df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Momentum indicators
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
        
        # Williams %R
        df['Williams_R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        
        # Volatility
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        df['Volatility_50'] = df['Close'].pct_change().rolling(window=50).std() * np.sqrt(252) * 100
        
        # Price position relative to moving averages
        df['Price_to_SMA_20'] = df['Close'] / df['SMA_20'] - 1
        df['Price_to_SMA_50'] = df['Close'] / df['SMA_50'] - 1
        df['Price_to_SMA_200'] = df['Close'] / df['SMA_200'] - 1
        
        # Golden/Death cross signal
        df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']
        
        # Drop rows with NaN
        initial_len = len(df)
        df.dropna(inplace=True)
        print(f"✅ Added {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} indicators")
        print(f"   Dropped {initial_len - len(df)} rows due to indicator warmup period")
        
        return df
    
    def prepare_sequences(self, df, target_col='Close', sequence_length=60, train_split=0.8):
        """Prepare sequences for LSTM."""
        print(f"📦 Creating sequences with {sequence_length}-day lookback...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        features = df[feature_cols].values
        target = df[[target_col]].values
        
        features_scaled = self.scaler.fit_transform(features)
        target_scaled = self.price_scaler.fit_transform(target)
        
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        split_idx = int(len(X) * train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.train_dates = df.index[sequence_length:split_idx + sequence_length]
        self.test_dates = df.index[split_idx + sequence_length:]
        
        print(f"✅ Train: {len(X_train)} | Test: {len(X_test)} | Shape: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def inverse_transform(self, scaled_values):
        """Convert scaled values back to original scale."""
        return self.price_scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()


class EnhancedLSTMModel:
    """Enhanced LSTM model with more features."""
    
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build(self, lstm_units=[256, 128, 64], dropout=0.3):
        """Build enhanced LSTM architecture."""
        print("🏗️ Building Enhanced LSTM Model...")
        
        self.model = Sequential([
            # First layer - Bidirectional for better context
            Bidirectional(LSTM(lstm_units[0], return_sequences=True),
                         input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(dropout),
            
            # Second LSTM layer
            LSTM(lstm_units[1], return_sequences=True),
            BatchNormalization(),
            Dropout(dropout),
            
            # Third LSTM layer
            LSTM(lstm_units[2], return_sequences=False),
            BatchNormalization(),
            Dropout(dropout),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Huber loss is more robust to outliers
            metrics=['mae', 'mse']
        )
        
        print("✅ Model built!")
        self.model.summary()
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              save_best=True, model_path='best_model.keras'):
        """Train with callbacks."""
        print(f"\n🚀 Training for up to {epochs} epochs...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        ]
        
        if save_best:
            callbacks.append(ModelCheckpoint(model_path, monitor='val_loss', 
                                            save_best_only=True, verbose=1))
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }


def create_visualizations(processor, y_train_orig, train_pred, y_test_orig, test_pred, 
                         history, test_metrics, ticker, output_dir='.'):
    """Generate all visualization plots."""
    
    # 1. Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', color='#3498DB', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#E74C3C', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE', color='#3498DB', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', color='#E74C3C', linewidth=2)
    axes[1].set_title('Training & Validation MAE', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Predictions Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Full timeline
    ax = axes[0, 0]
    ax.plot(processor.train_dates, y_train_orig, label='Train Actual', color='#2E86AB', alpha=0.7)
    ax.plot(processor.train_dates, train_pred, label='Train Predicted', color='#3498DB', linestyle='--', alpha=0.7)
    ax.plot(processor.test_dates, y_test_orig, label='Test Actual', color='#27AE60', linewidth=2)
    ax.plot(processor.test_dates, test_pred, label='Test Predicted', color='#E74C3C', linestyle='--', linewidth=2)
    ax.axvline(x=processor.test_dates[0], color='gray', linestyle=':', linewidth=2)
    ax.set_title(f'{ticker} - Complete Prediction Timeline', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price ($)')
    
    # Test zoom
    ax = axes[0, 1]
    ax.plot(processor.test_dates, y_test_orig, label='Actual', color='#27AE60', linewidth=2)
    ax.plot(processor.test_dates, test_pred, label='Predicted', color='#E74C3C', linewidth=2, linestyle='--')
    ax.fill_between(processor.test_dates, y_test_orig, test_pred, alpha=0.2, color='orange')
    ax.set_title('Test Period (Zoomed)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price ($)')
    
    # Scatter
    ax = axes[1, 0]
    ax.scatter(y_test_orig, test_pred, alpha=0.5, color='#9B59B6', s=30)
    min_val, max_val = min(y_test_orig.min(), test_pred.min()), max(y_test_orig.max(), test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 1]
    errors = y_test_orig - test_pred
    ax.hist(errors, bins=50, color='#3498DB', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=errors.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    ax.set_title('Prediction Error Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Error ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_display = {k: v for k, v in test_metrics.items()}
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
    bars = ax.barh(list(metrics_display.keys()), list(metrics_display.values()), color=colors)
    for bar, val in zip(bars, metrics_display.values()):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_title(f'{ticker} - Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved visualizations to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='LSTM Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='5y', help='Data period (1y, 2y, 5y, 10y, max)')
    parser.add_argument('--sequence', type=int, default=60, help='Sequence length (lookback days)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🔮 LSTM Stock Prediction for {args.ticker}")
    print("=" * 60)
    
    # Initialize
    processor = RealStockDataProcessor()
    
    # Fetch data
    df = processor.fetch_stock_data(args.ticker, args.period)
    df = processor.add_technical_indicators(df)
    
    # Prepare sequences
    X_train, X_test, y_train, y_test, features = processor.prepare_sequences(
        df, sequence_length=args.sequence
    )
    
    # Build and train model
    model = EnhancedLSTMModel(args.sequence, X_train.shape[2])
    model.build()
    history = model.train(X_train, y_train, X_test, y_test, 
                         epochs=args.epochs, batch_size=args.batch,
                         model_path=f'{args.output}/{args.ticker}_model.keras')
    
    # Predictions
    train_pred = processor.inverse_transform(model.predict(X_train))
    test_pred = processor.inverse_transform(model.predict(X_test))
    y_train_orig = processor.inverse_transform(y_train)
    y_test_orig = processor.inverse_transform(y_test)
    
    # Metrics
    test_metrics = EnhancedLSTMModel.calculate_metrics(y_test_orig, test_pred)
    
    print("\n📊 Test Set Performance:")
    for k, v in test_metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # Visualizations
    create_visualizations(processor, y_train_orig, train_pred, y_test_orig, test_pred,
                         history, test_metrics, args.ticker, args.output)
    
    print("\n✅ Complete!")
    print(f"   Model saved: {args.output}/{args.ticker}_model.keras")
    
    return model, processor, test_metrics


if __name__ == "__main__":
    main()
