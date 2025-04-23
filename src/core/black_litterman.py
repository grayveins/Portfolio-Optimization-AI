
import pandas as pd
from typing import Optional
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.risk_models import sample_cov


class BlackLittermanModelWrapper:
    """
    Wrapper around PyPortfolioOpt's Black-Litterman Model for easy integration.
    """

    def __init__(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series,
        views: pd.Series,
        omega: Optional[pd.DataFrame] = None,
        tau: float = 0.05
    ):
        """
        Initializes the Black-Litterman Model.

        :param cov_matrix: Covariance matrix of asset returns
        :param market_weights: Implied market capitalizations or beliefs (sum to 1)
        :param views: User or AI-generated expected returns
        :param omega: Uncertainty matrix (optional)
        :param tau: Scalar indicating uncertainty in the prior estimate
        """
        self.cov_matrix = cov_matrix
        self.market_weights = market_weights
        self.views = views
        self.omega = omega
        self.tau = tau

        self.bl = BlackLittermanModel(
            cov_matrix=self.cov_matrix,
            pi=None,
            absolute_views=self.views,
            omega=self.omega,
            market_caps=self.market_weights,
            tau=self.tau
        )

    def get_bl_returns(self) -> pd.Series:
        """
        Returns the adjusted expected returns from the Black-Litterman model.
        """
        return self.bl.bl_returns()

    def get_bl_cov(self) -> pd.DataFrame:
        """
        Returns the adjusted covariance matrix.
        """
        return self.bl.bl_cov()

    def get_all(self) -> tuple:
        """
        Returns both BL-adjusted returns and covariance matrix.
        """
        return self.get_bl_returns(), self.get_bl_cov()
