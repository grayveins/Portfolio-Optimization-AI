import logging
import yfinance as yf
from typing import List, Optional, Tuple
import pandas as pd


class DataLoader:
    """
    Handles fetching and preprocessing of stock price data using yfinance.
    Provides cleaned Adjusted Close data for valid tickers.
    """

    @staticmethod
    def get_data(
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "6mo",        # Default range: past 6 months
        frequency: str = "Adj Close",         # Type of price data to extract
        return_updated_tickers: bool = False  # Return cleaned ticker list too
    ) -> Tuple[pd.DataFrame, Optional[List[str]]]:
        """
        Downloads and filters historical stock data.

        Parameters:
        - tickers: List of asset tickers (e.g. ['AAPL', 'MSFT', 'GOOGL'])
        - start_date, end_date: Date range (if period is None)
        - period: Shorthand like '6mo', '1y', 'max'
        - frequency: Column to extract: 'Adj Close', 'Close', 'Open', etc.
        - return_updated_tickers: Whether to return cleaned/valid tickers list

        Returns:
        - DataFrame of price data (date-indexed)
        - Optional: list of valid tickers that returned data
        """

        valid_tickers = []

        # Step 1: Validate each ticker individually using a short 1mo test
        for ticker in tickers:
            try:
                hist = yf.download(ticker, period="1mo")
                if hist is not None and not hist.empty:
                    valid_tickers.append(ticker)
                else:
                    print(f"Ticker '{ticker}' is invalid or has no recent data.")
            except Exception as e:
                print(f"Error while validating ticker '{ticker}': {e}")

        # Step 2: Fail fast if no tickers were valid
        if not valid_tickers:
            raise ValueError("No valid tickers provided.")

        try:
            # Step 3: Download historical price data for all valid tickers
            if period:
                data = yf.download(valid_tickers, period=period, group_by="ticker", auto_adjust=False)
            else:
                if not start_date or not end_date:
                    raise ValueError("Start date and end date must be specified if not using period.")
                data = yf.download(valid_tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=False)

            # Step 4: Handle column extraction depending on number of tickers
            if len(valid_tickers) > 1:
                # Multi-index dataframe: (date, ticker, field)
                extracted = data.xs(frequency, axis=1, level=1)
            else:
                # Single ticker: flat dataframe
                extracted = data[[frequency]].rename(columns={frequency: valid_tickers[0]})

            # Step 5: Return data + optionally the cleaned ticker list
            return (extracted, valid_tickers) if return_updated_tickers else (extracted, None)

        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise
