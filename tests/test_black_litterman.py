import pandas as pd
import numpy as np
from src.core.black_litterman import BlackLittermanModelWrapper


def test_black_litterman_model():
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Simulated covariance matrix
    cov_matrix = pd.DataFrame([
        [0.04, 0.006, 0.008],
        [0.006, 0.03, 0.005],
        [0.008, 0.005, 0.035]
    ], index=tickers, columns=tickers)

    # Simulated market cap weights (these should sum to 1)
    market_weights = pd.Series([0.4, 0.3, 0.3], index=tickers)

    # Simulated views (your subjective or model-generated returns)
    views = pd.Series([0.11, 0.09, 0.10], index=tickers)

    # Initialize the Black-Litterman model
    bl_model = BlackLittermanModelWrapper(
        cov_matrix=cov_matrix,
        market_weights=market_weights,
        views=views
    )

    # Get adjusted returns and covariance
    bl_returns = bl_model.get_bl_returns()
    bl_cov = bl_model.get_bl_cov()

    # Assertions for sanity check
    assert isinstance(bl_returns, pd.Series), "Expected returns should be a Series"
    assert isinstance(bl_cov, pd.DataFrame), "Covariance matrix should be a DataFrame"

    print("Black-Litterman adjusted returns:")
    print(bl_returns)

    print("\nAdjusted Covariance Matrix:")
    print(bl_cov)


if __name__ == "__main__":
    test_black_litterman_model()
