import pandas as pd
import yfinance as yf
import numpy as np


class CapmCalculator:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def calculate_risk_free_rate(self):
        rf_data = yf.download("^IRX", start=self.start_date, end=self.end_date)

        if rf_data.empty:
            raise ValueError("No risk-free rate data found for the given period.")

        if "Adj Close" in rf_data.columns:
            return rf_data["Adj Close"].iloc[-1] / 100
        elif "Close" in rf_data.columns:
            return rf_data["Close"].iloc[-1] / 100
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in risk-free data.")

    def calculate_market_return(self):
        market_data = yf.download("^GSPC", start=self.start_date, end=self.end_date)
        if market_data.empty:
            raise ValueError("No market data found for ^GSPC.")
        if "Adj Close" in market_data.columns:
            close = market_data["Adj Close"]
        elif "Close" in market_data.columns:
            close = market_data["Close"]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in market data.")

        daily_returns = close.pct_change().dropna()
        return (1 + daily_returns.mean()) ** 252 - 1

    def calculate_beta(self, ticker):
        market_raw = yf.download("^GSPC", start=self.start_date, end=self.end_date)
        stock_raw = yf.download(ticker, start=self.start_date, end=self.end_date)

        if market_raw.empty or stock_raw.empty:
            raise ValueError(f"No data for market or ticker: {ticker}")

        if "Adj Close" in market_raw.columns:
            market_data = market_raw["Adj Close"]
        elif "Close" in market_raw.columns:
            market_data = market_raw["Close"]
        else:
            raise KeyError("No 'Adj Close' or 'Close' in market data.")

        if "Adj Close" in stock_raw.columns:
            stock_data = stock_raw["Adj Close"]
        elif "Close" in stock_raw.columns:
            stock_data = stock_raw["Close"]
        else:
            raise KeyError(f"No 'Adj Close' or 'Close' in stock data for {ticker}.")

        market_returns = market_data.pct_change().dropna()
        stock_returns = stock_data.pct_change().dropna()

        combined = pd.concat([market_returns, stock_returns], axis=1).dropna()

        cov = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1]
        var_market = np.var(combined.iloc[:, 0])

        return cov / var_market

    def calculate_expected_return(self, tickers):
        rf = self.calculate_risk_free_rate()
        rm = self.calculate_market_return()
        expected = {}

        for ticker in tickers:
            beta = self.calculate_beta(ticker)
            expected[ticker] = rf + beta * (rm - rf)

        return pd.Series(expected)
