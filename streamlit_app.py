import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from src.core.optimizer import PortfolioOptimizer
from src.core.black_litterman import BlackLittermanModelWrapper
from src.core.market_data import DataLoader
from pypfopt import expected_returns, risk_models
import plotly.express as px

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# Minimalist heading
st.markdown(
    """
    <style>
        .main {
            background-color: #191724;
            color: #e0def4;
        }
        .centered-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 600;
            color: #f6c177;
            margin-bottom: 0.5rem;
        }
        .subtle-subtitle {
            text-align: center;
            font-size: 1rem;
            font-weight: 300;
            color: #908caa;
            margin-top: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='centered-title'>Portfolio Optimization App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle-subtitle'>Simple, transparent, and data-driven investing</div>", unsafe_allow_html=True)


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
    forecast_source = st.radio("Forecast Source", ["Mean Historical Return", "GPT", "Capital Asset Pricing Model (CAPM)"])
    investment = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=500)

run = st.sidebar.button("Run Optimization")

# Main Title
st.title("S&P500 Optimizer")

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

    if len(valid) < 2 or prices.empty or prices.shape[0] < 5:
        st.error("Not enough valid tickers or price data. Try a different range or assets.")
        st.stop()

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # Forecast logic
    if forecast_source == "Mean Historical Return":
        views = mu.copy()

    elif forecast_source == "GPT":
        from src.ai.gpt_forecaster import GPTForecaster
        gpt = GPTForecaster()
        views_dict = {}
        for t in valid:
            try:
                forecast = gpt.generate_forecast(t)
                views_dict[t] = forecast.get("expected_return", mu[t])
            except:
                views_dict[t] = mu[t]
        views = pd.Series(views_dict)

    elif forecast_source == "Capital Asset Pricing Model (CAPM)":
        from src.core.expected_return import CapmCalculator
        capm = CapmCalculator(str(start_date), str(end_date))
        views = capm.calculate_expected_return(valid)

        if views.empty:
            st.warning("⚠️ CAPM returned no values. Falling back to historical mean.")
            views = mu.copy()

    # Clean up views
    views = views[~views.index.duplicated(keep="last")].dropna()
    market_weights = pd.Series([1 / len(views)] * len(views), index=views.index)

    # Final alignment
    common_assets = views.index.intersection(S.columns).intersection(S.index)
    views = views.loc[common_assets]
    S = S.loc[common_assets, common_assets]
    market_weights = market_weights.loc[common_assets]

    if len(common_assets) < 2:
        st.error("Not enough overlapping assets after alignment. Try adjusting tickers or date range.")
        st.stop()

    # Run Black-Litterman model
    bl_model = BlackLittermanModelWrapper(S, market_weights, views)
    bl_returns = bl_model.get_bl_returns()
    bl_cov = bl_model.get_bl_cov()

    if bl_returns.empty or bl_cov.empty:
        st.error("Black-Litterman output invalid. Try another forecast method.")
        st.stop()

    # Optimize
    optimizer = PortfolioOptimizer(expected_returns=bl_returns, cov_matrix=bl_cov)
    weights = optimizer.maximize_sharpe() if opt_method == "Maximize Sharpe" else optimizer.minimize_volatility()
    performance = optimizer.portfolio_performance(weights)

    # Allocation
    st.markdown("## Portfolio Allocation")
    col1, col2 = st.columns(2)
    alloc_df = weights.rename("Allocation").reset_index().rename(columns={"index": "Ticker"})
    alloc_df["Allocation %"] = (alloc_df["Allocation"] * 100).round(2)

    with col1:
        st.dataframe(alloc_df[["Ticker", "Allocation %"]])
    with col2:
        fig = px.pie(alloc_df, names="Ticker", values="Allocation", hole=0.4, title="Allocation Breakdown")
        st.plotly_chart(fig)

    # Performance
    st.markdown("## Portfolio Performance")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.metric("Expected Annual Return", f"{performance['expected_annual_return']:.2%}")
    perf_col2.metric("Annual Volatility", f"{performance['annual_volatility']:.2%}")
    perf_col3.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")

    # Growth chart
    st.markdown("## Portfolio Growth")
    normalized = prices[common_assets].div(prices[common_assets].iloc[0])
    weighted = normalized.mul(weights[common_assets], axis=1)
    growth = weighted.sum(axis=1) * investment

    fig = px.area(
        growth,
        labels={"index": "Date", "value": "Portfolio Balance ($)"},
        title="Simulated Portfolio Growth Over Time"
    )
    fig.update_traces(line_color="blue", fillcolor="rgba(0,0,255,0.2)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

