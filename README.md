# 🔮 LSTM Stock Price Prediction Model

A complete machine learning pipeline for stock price prediction using Long Short-Term Memory (LSTM) neural networks with TensorFlow/Keras.

## 📁 Files Included

| File | Description |
|------|-------------|
| `stock_prediction_lstm.py` | Complete standalone model with sample data generation |
| `stock_prediction_real.py` | Enhanced version with real Yahoo Finance data fetching |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Sample Data

```bash
python stock_prediction_lstm.py
```

This generates synthetic stock data and produces visualizations without needing internet access.

### 3. Run with Real Stock Data

```bash
# Default: Apple stock, 5 years of data
python stock_prediction_real.py

# Custom ticker and settings
python stock_prediction_real.py --ticker MSFT --period 3y --epochs 150

# Full options
python stock_prediction_real.py --ticker GOOGL --period 5y --sequence 90 --epochs 100 --batch 64 --output ./results
```

## 📊 Features

### Data Processing
- **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, Stochastic Oscillator, Williams %R, OBV, and more
- **Feature Engineering**: 40+ derived features for better prediction
- **Automatic Scaling**: MinMax normalization for optimal neural network performance

### Model Architecture
```
Bidirectional LSTM (256 units) → BatchNorm → Dropout
         ↓
    LSTM (128 units) → BatchNorm → Dropout
         ↓
    LSTM (64 units) → BatchNorm → Dropout
         ↓
    Dense (64) → Dense (32) → Output (1)
```

### Training Features
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Reduction**: Adaptive learning rate on plateaus
- **Model Checkpointing**: Saves best model during training
- **Huber Loss**: Robust to outliers in price data

### Visualization Outputs
1. **Training History**: Loss and MAE curves
2. **Prediction Timeline**: Full train/test comparison
3. **Scatter Plot**: Actual vs Predicted correlation
4. **Error Distribution**: Histogram of prediction errors
5. **Metrics Summary**: Visual performance metrics

## 📈 Model Metrics

The model evaluates using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ticker` | AAPL | Stock symbol |
| `--period` | 5y | Data period (1y, 2y, 5y, 10y, max) |
| `--sequence` | 60 | Lookback window in days |
| `--epochs` | 100 | Maximum training epochs |
| `--batch` | 32 | Training batch size |
| `--output` | . | Output directory for plots/models |

## 💡 Tips for Better Predictions

1. **More Data**: Use `--period 10y` or `max` for more training data
2. **Longer Sequences**: Try `--sequence 90` or `120` for more context
3. **Feature Selection**: Edit the code to include/exclude specific indicators
4. **Ensemble**: Train multiple models and average predictions
5. **Walk-Forward Validation**: Implement rolling window validation for production

## ⚠️ Disclaimer

This model is for **educational purposes only**. Stock price prediction is inherently uncertain and this model should NOT be used for actual trading decisions. Past performance does not guarantee future results.

## 📚 Technical Details

### Input Features (40+)
- Price data: Open, High, Low, Close, Volume
- Moving Averages: SMA (5, 10, 20, 50, 100, 200), EMA variants
- Momentum: MACD, RSI, Stochastic, Williams %R, ROC
- Volatility: ATR, Bollinger Bands, Historical Volatility
- Volume: OBV, Volume Ratio, Volume SMA

### Sequence Structure
```
Day 1-60 Features → Predict Day 61 Close Price
Day 2-61 Features → Predict Day 62 Close Price
...
```

## 🛠️ Extending the Model

### Add Custom Indicators
```python
# In add_technical_indicators method:
df['Custom_Indicator'] = your_calculation(df)
```

### Modify Architecture
```python
# In build method:
model.build(lstm_units=[512, 256, 128], dropout=0.4)
```

### Multi-Step Prediction
Modify the output layer to predict multiple future days.

---

Created with ❤️ for stock market enthusiasts and ML learners.
