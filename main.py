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
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

COIN_GECKO_API = "https://api.coingecko.com/api/v3"
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

crypto_models = {}

### 🚀 Fetch Historical Crypto Data with More Granularity ###
def fetch_crypto_history(symbol="bitcoin", days=365, interval="daily", split_seconds=None):
    url = f"{COIN_GECKO_API}/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": interval}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {"error": "Invalid response from API"}
    
    data = response.json()
    if "prices" not in data or "total_volumes" not in data:
        return {"error": "Invalid response data"}
    
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["volume"] = [x[1] for x in data["total_volumes"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Split timestamps if needed (e.g., create finer time samples within given data)
    if split_seconds:
        df = df.resample(f'{split_seconds}S', on='timestamp').interpolate()
    
    df["MA_7"] = df["price"].rolling(window=7).mean()
    df["MA_14"] = df["price"].rolling(window=14).mean()
    df["RSI"] = compute_rsi(df["price"], 14)
    df["MACD"], df["MACD_Signal"] = compute_macd(df["price"])
    df["Bollinger_Upper"], df["Bollinger_Lower"] = compute_bollinger_bands(df["price"])
    df["Volatility"] = df["price"].pct_change().rolling(7).std()
    df.fillna(method="bfill", inplace=True)
    return df

### 🏋️‍♂️ Train Improved LSTM Model with Batch Processing ###
def train_lstm_model(symbol="bitcoin", days=365, interval="daily", split_seconds=None, lookback=30, batch_size=512):
    df = fetch_crypto_history(symbol, days, interval, split_seconds)
    if "error" in df:
        return None, None, None
    
    scaler = MinMaxScaler()
    features = ["price", "volume", "MA_7", "MA_14", "RSI", "MACD", "Bollinger_Upper", "Volatility"]
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    model = LSTMModel(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(250):
        for i in range(0, len(X), batch_size):  # Process data in batches
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model, scaler, df

### 🧠 Improved LSTM Model with Dropout ###
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

### 🏋️‍♂️ Predict Crypto Prices with Hourly Granularity ###
@app.get("/predict/{symbol}/{days}")
def predict_crypto(symbol: str, days: int = 7):
    if symbol not in crypto_models:
        model, scaler, data = train_lstm_model(symbol)
        if model is None:
            return {"error": "Unable to fetch data for this cryptocurrency."}
        crypto_models[symbol] = (model, scaler, data)
    
    model, scaler, crypto_data = crypto_models[symbol]
    sentiment_score = fetch_crypto_sentiment(symbol)
    last_days = crypto_data.iloc[-30:][["price", "volume", "MA_7", "MA_14", "RSI", "MACD", "Bollinger_Upper", "Volatility"]].values
    scaled_last_days = scaler.transform(last_days)
    input_seq = torch.tensor(scaled_last_days, dtype=torch.float32).reshape(1, 30, -1)
    
    predictions = []
    for i in range(days):
        day_predictions = []
        for hour in range(24):
            with torch.no_grad():
                predicted_scaled = model(input_seq).item()
            predicted_price = scaler.inverse_transform([[predicted_scaled] + [0] * 7])[0, 0]
            adjusted_price = predicted_price * (1 + sentiment_score * 0.1)
            timestamp = datetime.utcnow() + timedelta(days=i, hours=hour)
            day_predictions.append({"datetime": str(timestamp), "predicted_price": round(adjusted_price, 2)})
        predictions.extend(day_predictions)
    
    return predictions

### 📈 Compute RSI ###
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

### 📊 Compute MACD ###
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

### 📉 Compute Bollinger Bands ###
def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


### 📰 Fetch Crypto Sentiment ###
def fetch_crypto_sentiment(symbol="bitcoin"):
    params = {"q": symbol, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt"}
    response = requests.get(NEWS_API_URL, params=params)
    
    if response.status_code != 200:
        return 0  # Neutral sentiment
    
    articles = response.json().get("articles", [])
    headlines = [article["title"] for article in articles]
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiments = [sentiment_analyzer(headline)[0]["label"] for headline in headlines]
    
    positive = sentiments.count("POSITIVE")
    negative = sentiments.count("NEGATIVE")
    
    return (positive - negative) / max(len(sentiments), 1)
