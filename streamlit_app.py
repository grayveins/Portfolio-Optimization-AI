import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from src.core.optimizer import PortfolioOptimizer
from src.core.black_litterman import BlackLittermanModelWrapper
from src.core.market_data import DataLoader
from pypfopt import expected_returns, risk_models
import plotly.express as px

# Load S&P 500 tickers
sp500_df = pd.read_csv("data/sp500_tickers.csv")
sp500_df = sp500_df[["Ticker", "Name"]].dropna().drop_duplicates(subset="Ticker")
ticker_dict = sp500_df.set_index("Ticker")["Name"].to_dict()

# Sidebar
st.sidebar.title("Portfolio Optimizer")
tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=sp500_df["Ticker"],
    default=["AAPL", "MSFT", "GOOGL"],
    format_func=lambda x: f"{x} - {ticker_dict.get(x, '')}"
)

# Time horizon
today = date.today()
st.sidebar.markdown("### Time Horizon")
start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1), min_value=date(2010, 1, 1), max_value=today)
end_date = st.sidebar.date_input("End Date", value=today, min_value=start_date, max_value=today)
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Forecast and optimizer
with st.sidebar.expander("Forecast & Optimization Settings"):
    opt_method = st.radio("Optimization Method", ["Maximize Sharpe", "Minimize Volatility"])
    forecast_source = st.radio("Forecast Source", ["Mean Historical Return", "Mock GPT", "CAPM (placeholder)"])
    investment = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=500)

run = st.sidebar.button("Run Optimization")

# Main Title
st.title("GPT Full Port")

if run:
    if len(tickers) < 2:
        st.warning("Please select at least two tickers.")
        st.stop()

    loader = DataLoader()
    prices, valid = loader.get_data(
        tickers,
        start_date=str(start_date),
        end_date=str(end_date),
        frequency="Adj Close",
        return_updated_tickers=True
    )

    if len(valid) < 2:
        st.warning("Not enough valid tickers to proceed.")
        st.stop()
    if prices.empty or prices.shape[0] < 5:
        st.error("Not enough historical price data for the selected time range. Try expanding the range.")
        st.stop()

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    if forecast_source == "Mean Historical Return":
        views = mu.copy()
    else:
        mock_views = {t: {"expected_return": mu[t], "confidence": 70} for t in valid}
        views = pd.Series({t: mock_views[t]["expected_return"] for t in valid})

    market_weights = pd.Series([1/len(valid)] * len(valid), index=valid)

    bl_model = BlackLittermanModelWrapper(S, market_weights, views)
    bl_returns = bl_model.get_bl_returns()
    bl_cov = bl_model.get_bl_cov()

    optimizer = PortfolioOptimizer(expected_returns=bl_returns, cov_matrix=bl_cov)
    weights = optimizer.maximize_sharpe() if opt_method == "Maximize Sharpe" else optimizer.minimize_volatility()
    performance = optimizer.portfolio_performance(weights)

    # Allocation
    st.markdown("## Portfolio Allocation\n")
    col1, col2 = st.columns(2)
    alloc_df = weights.rename("Allocation").reset_index().rename(columns={"index": "Ticker"})
    alloc_df["Allocation %"] = (alloc_df["Allocation"] * 100).round(2)

    with col1:
        st.dataframe(alloc_df[["Ticker", "Allocation %"]])

    with col2:
        fig = px.pie(alloc_df, names="Ticker", values="Allocation", hole=0.4, title="Allocation Breakdown")
        st.plotly_chart(fig)

    # Performance
    st.markdown("## Portfolio Performance\n")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.metric("Expected Annual Return", f"{performance['expected_annual_return']:.2%}")
    perf_col2.metric("Annual Volatility", f"{performance['annual_volatility']:.2%}")
    perf_col3.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")

    # Projected Return
    projected_gain = investment * performance['expected_annual_return']
    st.markdown("## Projected Return\n")
    st.write(f"Based on your investment of ${investment:,.0f}, your projected return is ${projected_gain:,.2f} after one year.")

    # Simulated portfolio value over time
    st.markdown("## Portfolio Growth\n")
    normalized_prices = prices[valid].div(prices[valid].iloc[0])
    weighted_prices = normalized_prices.mul(weights[valid], axis=1)
    portfolio_growth = weighted_prices.sum(axis=1) * investment

    fig_growth = px.area(
        portfolio_growth,
        labels={"index": "Date", "value": "Portfolio Balance ($)"},
        title="Simulated Portfolio Growth Over Time"
    )
    fig_growth.update_traces(line_color="blue", fillcolor="rgba(0,0,255,0.2)")
    fig_growth.update_layout(showlegend=False)
    st.plotly_chart(fig_growth)

