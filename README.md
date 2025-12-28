# 📈 LSTM Stock & Crypto Price Prediction

A deep learning model using LSTM (Long Short-Term Memory) neural networks to predict stock and cryptocurrency prices.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Real-time data fetching** from Yahoo Finance (stocks, crypto, ETFs)
- **40+ technical indicators** (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, etc.)
- **Bidirectional LSTM** architecture with dropout regularization
- **Early stopping** and learning rate scheduling
- **Comprehensive visualizations** (predictions, training history, error analysis)

## 🛠️ Installation

### Prerequisites
- Python 3.11 (TensorFlow doesn't support Python 3.12+ on Windows yet)

### Setup
```bash
# Clone the repository
git clone https://github.com/Virus-101/stock-pred.git
cd stock-pred

# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Predict Bitcoin Prices
```bash
python stock_prediction_real.py --ticker BTC-USD --period max --epochs 150
```

### Predict Stock Prices
```bash
# Apple
python stock_prediction_real.py --ticker AAPL --period 5y --epochs 100

# Tesla
python stock_prediction_real.py --ticker TSLA --period 5y --epochs 100

# Microsoft
python stock_prediction_real.py --ticker MSFT --period 5y --epochs 100
```

### Run with Sample Data (No Internet Required)
```bash
python stock_prediction_lstm.py
```

### Command Line Options
| Option | Default | Description |
|--------|---------|-------------|
| `--ticker` | AAPL | Stock/crypto symbol (e.g., BTC-USD, AAPL, TSLA) |
| `--period` | 5y | Data period: 1y, 2y, 5y, 10y, max |
| `--sequence` | 60 | Lookback window in days |
| `--epochs` | 100 | Maximum training epochs |
| `--batch` | 32 | Training batch size |

## 📊 Model Architecture
```
Input (60 days × 40+ features)
          ↓
Bidirectional LSTM (256 units)
          ↓
    BatchNorm + Dropout (0.3)
          ↓
     LSTM (128 units)
          ↓
    BatchNorm + Dropout (0.3)
          ↓
     LSTM (64 units)
          ↓
    BatchNorm + Dropout (0.3)
          ↓
    Dense (64) → Dense (32) → Output (1)
```

## 📈 Technical Indicators

- **Moving Averages**: SMA (5, 10, 20, 50, 100, 200), EMA
- **Momentum**: MACD, RSI, Stochastic Oscillator, Williams %R, ROC
- **Volatility**: ATR, Bollinger Bands, Historical Volatility
- **Volume**: OBV, Volume Ratio, Volume SMA

## 📁 Project Structure
```
stock-pred/
├── stock_prediction_lstm.py   # Model with sample data
├── stock_prediction_real.py   # Model with Yahoo Finance data
├── requirements.txt           # Python dependencies
└── README.md
```

## ⚠️ Disclaimer

**This project is for educational purposes only.**

Stock and cryptocurrency markets are inherently unpredictable. This model learns patterns from historical data but **cannot reliably predict future prices**.

**Do NOT use this model for real trading or investment decisions.**

## 📝 License

MIT License - feel free to use and modify for your own projects.

---

Built with ❤️ using TensorFlow and Python
