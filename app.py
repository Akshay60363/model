from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime, timedelta

app = Flask(_name)  # Corrected to __name

# Endpoint 1: NYSE U.S. 100 Index Price Data
@app.route('/get_nyse_data', methods=['GET'])
def get_nyse_data():
    try:
        # Get parameters from the request
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Convert to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Fetch NYSE Composite Index data
        ticker = "^NYA"
        data = yf.download(ticker, start=start_date, end=end_date)
        data_close = data['Close']
        data_close.index = data_close.index.astype(str)

        # Return the data as a JSON response
        return jsonify(data_close.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Endpoint 2: Cumulative sum of price movement for the given list of stocks
@app.route('/get_cumulative_movement', methods=['GET'])
def get_cumulative_movement():
    try:
        # Get parameters from the request
        stocks = request.args.get('stocks').split(',')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Convert to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        cumulative_price = None
        
        for stock in stocks:
            # Fetch stock data
            data = yf.download(stock, start=start_date, end=end_date)
            
            # Calculate cumulative sum of closing prices
            if cumulative_price is None:
                cumulative_price = data['Close'].to_numpy()
            else:
                cumulative_price += data['Close'].to_numpy()
        
        # Create DataFrame for the result
        cumulative_price_df = pd.DataFrame(cumulative_price, index=data.index, columns=["Cumulative Close"])
        cumulative_price_df.index = cumulative_price_df.index.astype(str)

        return jsonify(cumulative_price_df.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Endpoint 3: Random Forest Classifier for Buy/Sell/Hold Labels
@app.route('/generate_signals', methods=['GET'])
def generate_signals():
    try:
        # Get parameters from the request
        stocks = request.args.get('stocks').split(',')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Convert to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        signals = {}
        
        for stock in stocks:
            # Fetch stock data
            data = yf.download(stock, start=start_date, end=end_date)
            
            # Create features for the Random Forest
            data['Returns'] = data['Adj Close'].pct_change()
            data['Moving Average'] = data['Adj Close'].rolling(window=20).mean()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # Drop NaN values
            data = data.dropna()

            # Prepare training data
            X = data[['Returns', 'Moving Average', 'Volatility']]
            y = (data['Returns'] > 0).astype(int)
            
            # Train the Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)

            # Generate signals for every 3 months
            signals[stock] = []
            current_date = start_date
            while current_date < end_date:
                period_end = current_date + timedelta(days=90)
                period_data = data[(data.index >= current_date.strftime('%Y-%m-%d')) & 
                                   (data.index < period_end.strftime('%Y-%m-%d'))]
                
                if not period_data.empty:
                    X_period = period_data[['Returns', 'Moving Average', 'Volatility']]
                    predicted = clf.predict(X_period)
                    signal = 'Buy' if predicted.mean() > 0.7 else ('Hold' if predicted.mean() > 0.4 else 'Sell')
                    signals[stock].append({'date': current_date.strftime('%Y-%m-%d'), 'signal': signal})
                
                current_date = period_end

        return jsonify(signals)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if _name_ == '_main_':
    app.run(debug=True)