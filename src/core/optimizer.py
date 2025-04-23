import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

class PortfolioOptimizer:
    """
    Optimizes a portfolio based on expected returns and covariance matrix using PyPortfolioOpt.
    Supports maximizing Sharpe ratio and minimizing volatility.
    """

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the optimizer with expected returns and covariance matrix.

        :param expected_returns: Expected annual returns as a pandas Series
        :param cov_matrix: Covariance matrix of asset returns
        :param risk_free_rate: Risk-free rate used for Sharpe ratio calculation
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate

    def maximize_sharpe(self) -> pd.Series:
        """
        Optimize portfolio to maximize the Sharpe ratio.

        :return: Cleaned weights as a pandas Series
        """
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        cleaned_weights = ef.clean_weights()
        return pd.Series(cleaned_weights)

    def minimize_volatility(self) -> pd.Series:
        """
        Optimize portfolio to minimize total volatility.

        :return: Cleaned weights as a pandas Series
        """
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        return pd.Series(cleaned_weights)

    def portfolio_performance(self, weights: dict) -> dict:
        """
        Calculate expected performance metrics for a given set of weights.

        :param weights: Dictionary or Series of asset weights
        :return: Dict containing expected return, volatility, and Sharpe ratio
        """
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        ef.set_weights(weights)
        perf = ef.portfolio_performance(verbose=False)
        return {
            "expected_annual_return": perf[0],
            "annual_volatility": perf[1],
            "sharpe_ratio": perf[2]
        }
