import pandas as pd
import numpy as np
from src.core.optimizer import PortfolioOptimizer


def test_maximize_sharpe():
    # Simulated expected returns and covariance matrix
    tickers = ["AAPL", "MSFT", "GOOGL"]
    mu = pd.Series([0.12, 0.10, 0.11], index=tickers)

    cov_matrix = pd.DataFrame([
        [0.04, 0.006, 0.008],
        [0.006, 0.03, 0.005],
        [0.008, 0.005, 0.035]
    ], index=tickers, columns=tickers)

    # Instantiate the optimizer
    optimizer = PortfolioOptimizer(expected_returns=mu, cov_matrix=cov_matrix)

    # Optimize for maximum Sharpe ratio
    weights = optimizer.maximize_sharpe()

    # Assertions
    assert isinstance(weights, pd.Series), "Weights should be a pandas Series"
    assert np.isclose(weights.sum(), 1, atol=1e-3), "Weights should sum to 1"
    assert all(0 <= w <= 1 for w in weights), "Weights should be between 0 and 1"

    print("Max Sharpe weights:")
    print(weights)

    # Portfolio performance
    perf = optimizer.portfolio_performance(weights)
    print("Performance:")
    print(perf)


if __name__ == "__main__":
    test_maximize_sharpe()
