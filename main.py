import os
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
from datetime import datetime, timedelta
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Secure API keys from .env file
COIN_GECKO_API = "https://api.coingecko.com/api/v3"

### 🚀 Fetch Historical Crypto Data with New Indicators ###
def fetch_crypto_history(symbol="bitcoin", days=365):
    """Fetch historical crypto price & volume from CoinGecko API"""
    url = f"{COIN_GECKO_API}/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "prices" not in data or "total_volumes" not in data:
        return {"error": "Invalid response from API"}
    
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["volume"] = [x[1] for x in data["total_volumes"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Calculate Technical Indicators
    df["MA_7"] = df["price"].rolling(window=7).mean()  # 7-day moving average
    df["MA_14"] = df["price"].rolling(window=14).mean()  # 14-day moving average
    df["RSI"] = compute_rsi(df["price"], 14)  # RSI (Relative Strength Index)
    df["MACD"], df["MACD_Signal"] = compute_macd(df["price"])  # MACD Indicator
    df["Bollinger_Upper"], df["Bollinger_Lower"] = compute_bollinger_bands(df["price"])  # Bollinger Bands
    
    df.fillna(method="bfill", inplace=True)  # Fill missing values  
    return df

### 📈 Compute RSI (Relative Strength Index) ###
def compute_rsi(series, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

### 📊 Compute MACD Indicator ###
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD and MACD Signal Line"""
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

### 📉 Compute Bollinger Bands ###
def compute_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

### 🧠 Define LSTM Model ###
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

### 🏋️‍♂️ Train LSTM Model with More Data ###
def train_lstm_model(symbol="bitcoin", days=365, lookback=30):
    df = fetch_crypto_history(symbol, days)

    # Normalize prices & indicators
    scaler = MinMaxScaler()
    features = ["price", "volume", "MA_7", "MA_14", "RSI", "MACD", "Bollinger_Upper"]
    scaled_data = scaler.fit_transform(df[features])

    # Prepare dataset
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])  # Predicting price

    X, y = np.array(X), np.array(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    model = LSTMModel(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(150):  # Train for more epochs
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model, scaler, df

# Train model on startup
lstm_model, lstm_scaler, crypto_data = train_lstm_model()

@app.get("/predict/{symbol}/{days}")
def predict_crypto(symbol: str, days: int = 7):
    """Predict future crypto prices using LSTM with Extended Training"""
    global lstm_model, lstm_scaler, crypto_data

    lookback = 30
    last_days = crypto_data[["price", "volume", "MA_7", "MA_14", "RSI", "MACD", "Bollinger_Upper"]].values[-lookback:]

    scaled_last_days = lstm_scaler.transform(last_days)
    input_seq = torch.tensor(scaled_last_days, dtype=torch.float32).reshape(1, lookback, -1)

    future_predictions = []
    for i in range(days):
        with torch.no_grad():
            predicted_scaled = lstm_model(input_seq).item()
        
        predicted_price = lstm_scaler.inverse_transform([[predicted_scaled, 0, 0, 0, 0, 0, 0]])[0, 0]
        future_predictions.append({"date": str(datetime.utcnow() + timedelta(days=i+1)), "predicted_price": round(predicted_price, 2)})

        input_seq = torch.roll(input_seq, shifts=-1, dims=1)
        input_seq[0, -1, 0] = predicted_scaled  

    return future_predictions

### 📡 Fetch Live Crypto Prices ###
@app.get("/crypto/{symbol}")
def get_crypto_price(symbol: str):
    """Fetch real-time cryptocurrency price"""
    url = f"{COIN_GECKO_API}/simple/price"
    params = {"ids": symbol, "vs_currencies": "usd"}
    response = requests.get(url, params=params)
    return response.json()

### 🌍 Root Endpoint ###
@app.get("/")
def read_root():
    return {"message": "Enhanced Crypto AI Prediction API is running!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Use the PORT environment variable from Render
    uvicorn.run(app, host="0.0.0.0", port=port)