from src.core.market_data import DataLoader


def test_data_loader():
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "FAKE"]  # 'FAKE' should trigger error handling
    try:
        data, valid = DataLoader.get_data(
            tickers=tickers,
            period="6mo",
            frequency="Adj Close",
            return_updated_tickers=True
        )

        print(f"\nValid tickers found: {valid}")
        print(f"\nSample data (head):\n{data.head()}")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_data_loader()
