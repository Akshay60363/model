import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load pre-trained models and scaler
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("buy_sell_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return scaler, model, label_encoder

scaler, model, label_encoder = load_models()

# App Title
st.title("Stock Analysis and Signal Generation")

# Section 1: Fetch NYSE Data
st.header("1. NYSE U.S. 100 Index Price Data")

# Input form for NYSE data
with st.form("nyse_form"):
    start_date_nyse = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date_nyse = st.date_input("End Date", datetime.now())
    submitted_nyse = st.form_submit_button("Fetch NYSE Data")

if submitted_nyse:
    try:
        ticker = "^NYA"
        data = yf.download(ticker, start=start_date_nyse, end=end_date_nyse)
        st.write("### NYSE Close Prices", data[['Close']])
        st.line_chart(data['Close'])
    except Exception as e:
        st.error(f"Error fetching NYSE data: {e}")

# Section 2: Cumulative Price Movement
st.header("2. Cumulative Price Movement for Stocks")

# Input form for cumulative movement
with st.form("cumulative_form"):
    stocks_input = st.text_input("Enter Stock Tickers (comma-separated)", "AAPL,GOOG")
    start_date_cum = st.date_input("Start Date (Cumulative)", datetime.now() - timedelta(days=365))
    end_date_cum = st.date_input("End Date (Cumulative)", datetime.now())
    submitted_cum = st.form_submit_button("Calculate Cumulative Movement")

if submitted_cum:
    try:
        stock_list = stocks_input.split(',')
        cumulative_price = None

        for stock in stock_list:
            data = yf.download(stock, start=start_date_cum, end=end_date_cum)
            if cumulative_price is None:
                cumulative_price = data['Close']
            else:
                cumulative_price += data['Close']

        cumulative_df = pd.DataFrame(cumulative_price, columns=["Cumulative Close"])
        st.write("### Cumulative Price Movement", cumulative_df)
        st.line_chart(cumulative_df)
    except Exception as e:
        st.error(f"Error calculating cumulative price movement: {e}")

# Section 3: Generate Signals
st.header("3. Generate Buy/Sell/Hold Signals")

# Input form for signal generation
with st.form("signals_form"):
    stocks_signal = st.text_input("Enter Stock Tickers for Signals (comma-separated)", "AAPL,GOOG")
    start_date_signal = st.date_input("Start Date (Signals)", datetime.now() - timedelta(days=365))
    end_date_signal = st.date_input("End Date (Signals)", datetime.now())
    submitted_signal = st.form_submit_button("Generate Signals")

if submitted_signal:
    try:
        stock_list = stocks_signal.split(',')
        signals = {}

        for stock in stock_list:
            data = yf.download(stock, start=start_date_signal, end=end_date_signal)
            data['Returns'] = data['Adj Close'].pct_change()
            data['Moving Average'] = data['Adj Close'].rolling(window=20).mean()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data = data.dropna()

            # Scale the features
            features = ['Returns', 'Moving Average', 'Volatility']
            X = data[features]
            X_scaled = scaler.transform(X)

            # Predict signals
            predicted = model.predict(X_scaled)
            predicted_labels = label_encoder.inverse_transform(predicted)

            stock_signals = [
                {'date': idx.strftime('%Y-%m-%d'), 'signal': label}
                for idx, label in zip(data.index, predicted_labels)
            ]

            signals[stock] = stock_signals

        # Display signals
        for stock, stock_signals in signals.items():
            st.write(f"### Signals for {stock}")
            st.table(pd.DataFrame(stock_signals))

    except Exception as e:
        st.error(f"Error generating signals: {e}")
