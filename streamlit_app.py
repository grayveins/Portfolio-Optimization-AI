import streamlit as st
import pandas as pd
import numpy as np
from src.core.optimizer import PortfolioOptimizer
from src.core.black_litterman import BlackLittermanModelWrapper

# Mock forecast data (replace with GPT or CAPM integration)
mock_views = {
    "AAPL": {"expected_return": 0.08, "confidence": 75},
    "MSFT": {"expected_return": 0.07, "confidence": 70},
    "GOOGL": {"expected_return": 0.09, "confidence": 80},
}

# Sidebar
st.sidebar.title("Portfolio Optimizer")
tickers = st.sidebar.multiselect("Select Tickers", list(mock_views.keys()), default=list(mock_views.keys()))
opt_method = st.sidebar.radio("Optimization Method", ["Maximize Sharpe", "Minimize Volatility"])
forecast_source = st.sidebar.radio("Forecast Source", ["Mock GPT", "CAPM (placeholder)"])
run = st.sidebar.button("Run Optimization")

# Main Title
st.title("AI-Powered Portfolio Optimization")

if run:
    st.subheader("Selected Tickers")
    st.write(tickers)

    if len(tickers) < 2:
        st.warning("Please select at least two tickers.")
        st.stop()

    # Simulate price data
    cov_matrix = pd.DataFrame([
        [0.04, 0.006, 0.008],
        [0.006, 0.03, 0.005],
        [0.008, 0.005, 0.035]
    ], index=tickers, columns=tickers)

    views = pd.Series({t: mock_views[t]["expected_return"] for t in tickers})
    market_weights = pd.Series([1/len(tickers)] * len(tickers), index=tickers)  # Equal weight

    # Apply Black-Litterman
    bl_model = BlackLittermanModelWrapper(cov_matrix, market_weights, views)
    bl_returns = bl_model.get_bl_returns()
    bl_cov = bl_model.get_bl_cov()

    # Optimize portfolio
    optimizer = PortfolioOptimizer(expected_returns=bl_returns, cov_matrix=bl_cov)
    weights = optimizer.maximize_sharpe() if opt_method == "Maximize Sharpe" else optimizer.minimize_volatility()
    performance = optimizer.portfolio_performance(weights)

    # Display Results
    st.subheader("Optimized Weights")
    st.bar_chart(weights)

    st.subheader("Portfolio Performance")
    st.metric("Expected Annual Return", f"{performance['expected_annual_return']:.2%}")
    st.metric("Annual Volatility", f"{performance['annual_volatility']:.2%}")
    st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")

    st.subheader("Forecasted Returns (Mock Data)")
    st.json({t: mock_views[t] for t in tickers})
