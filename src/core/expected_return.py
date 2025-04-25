import yfinance as yf
import pandas as pd
import numpy as np


class CapmCalculator:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def calculate_risk_free_rate(self):
        rf_data = yf.download("^IRX", start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
        prices = self._extract_adj_close(rf_data, "^IRX")
        return prices.dropna().iloc[-1] / 100

    def calculate_market_return(self):
        market_data = yf.download("^GSPC", start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
        prices = self._extract_adj_close(market_data, "^GSPC")
        returns = prices.pct_change().dropna()
        return (1 + returns.mean()) ** 252 - 1

    def calculate_historical_return(self, ticker):
        data = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
        prices = self._extract_adj_close(data, ticker)
        returns = prices.pct_change().dropna()
        return (1 + returns.mean()) ** 252 - 1

    def calculate_beta(self, ticker):
        stock = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
        market = yf.download("^GSPC", start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)

        stock_prices = self._extract_adj_close(stock, ticker).resample("M").last().pct_change().dropna()
        market_prices = self._extract_adj_close(market, "^GSPC").resample("M").last().pct_change().dropna()

        aligned = pd.concat([stock_prices, market_prices], axis=1).dropna()
        aligned.columns = ["stock", "market"]

        if aligned.shape[0] < 3:
            raise ValueError(f"Not enough overlapping return data for {ticker}")

        cov = np.cov(aligned["stock"], aligned["market"])[0, 1]
        var_market = np.var(aligned["market"])

        beta = cov / var_market
        if np.isnan(beta):
            raise ValueError(f"Beta is NaN for {ticker}")
        return beta

    def calculate_expected_return(self, tickers):
        rf = self.calculate_risk_free_rate()
        rm = self.calculate_market_return()

        expected = {}
        for ticker in tickers:
            try:
                beta = self.calculate_beta(ticker)
                expected[ticker] = rf + beta * (rm - rf)
            except Exception as e:
                print(f"CAPM error for {ticker}: {e}")
                try:
                    expected[ticker] = self.calculate_historical_return(ticker)
                    print(f"Fallback: using historical return for {ticker}")
                except Exception as fallback_e:
                    print(f"Fallback failed for {ticker}: {fallback_e}")
                    expected[ticker] = np.nan

        return pd.Series(expected).dropna()

    def _extract_adj_close(self, df, ticker): 
        """Handles both single and multi-index yfinance DataFrames, with fallbacks for ^IRX and ^GSPC."""
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker}")

        # Handle multi-index structure
        if isinstance(df.columns, pd.MultiIndex):
            for col in ["Adj Close", "Close"]:
                if (col, ticker) in df.columns:
                    return df[(col, ticker)].dropna()
                elif col in df.columns.get_level_values(0):
                    return df[col].iloc[:, 0].dropna()
            raise ValueError(f"Neither 'Adj Close' nor 'Close' found for {ticker} in multi-index") 
        # Handle flat structure
        for col in ["Adj Close", "Close"]:
            if col in df.columns:
                return df[col].dropna()

        raise ValueError(f"Neither 'Adj Close' nor 'Close' found for {ticker}")
