import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np # Will need for MPT calculations
from scipy.optimize import minimize # For efficient frontier calculation

# --- Page Functions ---

def intro_mpt_page():
    st.header("Introduction to Modern Portfolio Theory (MPT)")
    st.write("""
    Modern Portfolio Theory (MPT) is a framework for constructing investment portfolios to maximize expected return for a given level of market risk, or minimize risk for a given level of expected return. It was pioneered by Harry Markowitz in his 1952 essay "Portfolio Selection."

    Key concepts of MPT include:
    - **Risk and Return:** Investors are assumed to be rational and risk-averse, meaning they prefer higher returns for the same risk, or lower risk for the same return.
    - **Diversification:** Combining different assets in a portfolio can reduce overall risk without sacrificing expected returns, especially if the assets are not perfectly positively correlated.
    - **Efficient Frontier:** This is a set of optimal portfolios that offer the highest expected return for a defined level of risk, or the lowest risk for a given level of expected return.
    - **Optimal Portfolio:** For a given investor, the optimal portfolio on the efficient frontier depends on their individual risk tolerance.
    """)

def math_foundation_page():
    st.header("Mathematical Foundation of MPT")

    # Section 1: Expected Return
    st.markdown(r"""
    **1. Expected Return of a Portfolio ($R_p$):**
    The expected return of a portfolio is the weighted average of the expected returns of the individual assets in the portfolio.
    """)
    st.latex(r'E(R_p) = \sum_{i=1}^{n} w_i E(R_i)')
    st.markdown(r"""
    Where:
    - $E(R_p)$ = Expected return of the portfolio
    - $w_i$ = Weight (proportion) of asset $i$ in the portfolio
    - $E(R_i)$ = Expected return of asset $i$
    - $n$ = Number of assets in the portfolio
    """)

    # Section 2: Variance
    st.markdown(r"""
    **2. Variance of a Portfolio ($\sigma_p^2$):**
    The variance of a portfolio measures its overall risk. It considers not only the variance of individual assets but also the covariance between them.
    """)
    st.latex(r'\sigma_p^2 = \sum_{i=1}^{n} w_i^2 \sigma_i^2 + \sum_{i=1}^{n} \sum_{j=1, i \neq j}^{n} w_i w_j \sigma_{ij}')
    st.markdown(r"""
    Where:
    - $\sigma_p^2$ = Variance of the portfolio
    - $\sigma_i^2$ = Variance of asset $i$
    - $\sigma_{ij}$ = Covariance between asset $i$ and asset $j$
    """)

    # Section 3: Standard Deviation
    st.markdown(r"""
    **3. Standard Deviation of a Portfolio ($\sigma_p$):**
    The standard deviation is the square root of the variance and is a common measure of portfolio volatility (risk).
    """)
    st.latex(r'\sigma_p = \sqrt{\sigma_p^2}')

    # Section 4: Sharpe Ratio
    st.markdown(r"""
    **4. Sharpe Ratio:**
    The Sharpe Ratio measures the risk-adjusted return of a portfolio. It indicates the amount of excess return (above the risk-free rate) per unit of risk (standard deviation).
    """)
    st.latex(r'\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}')
    st.markdown(r"""
    Where:
    - $R_f$ = Risk-free rate
    """)

    # Section 5: Efficient Frontier
    st.subheader("5. Efficient Frontier Computation")
    st.markdown(r"""
    The Efficient Frontier is not just a visual edge of random simulations; it's a curve that is calculated by solving a series of optimization problems. For each level of target return, we find the portfolio with the minimum possible risk (variance).

    **Matrix Notation:**
    To express the optimization problem more concisely, we can use matrix notation:
    - Let $\mathbf{w}$ be the vector of portfolio weights: $\mathbf{w} = [w_1, w_2, ..., w_n]^T$
    - Let $\mathbf{R}$ be the vector of expected returns: $\mathbf{R} = [E(R_1), E(R_2), ..., E(R_n)]^T$
    - Let $\mathbf{\Sigma}$ be the covariance matrix of returns.

    The expected return of the portfolio is then $E(R_p) = \mathbf{w}^T \mathbf{R}$, and the portfolio variance is $\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$.

    **Objective Function:**
    The goal is to minimize the portfolio variance:
    """)
    st.latex(r'\text{Minimize } \sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}')

    st.markdown(r"""
    **Constraints:**
    The optimization is subject to the following constraints:
    """)
    st.markdown(r"1. **The sum of the weights of all assets in the portfolio must be 1:**")
    st.latex(r'\mathbf{w}^T \mathbf{1} = 1')
    st.markdown(r"2. **The expected return of the portfolio must equal a target return $E(R_{\text{target}})$:**")
    st.latex(r'\mathbf{w}^T \mathbf{R} = E(R_{\text{target}})')
    st.markdown(r"3. **The weights of all assets must be non-negative (no short selling):**")
    st.latex(r'w_i \geq 0 \quad \forall i \in \{1, ..., n\}')

def step_by_step_page():
    st.header("Step-by-Step Guide: Math & Code")

    st.markdown("""
    This page walks through the mathematical concepts of Modern Portfolio Theory and shows the corresponding Python code used to perform the calculations.
    """)

    if 'all_stock_data' not in st.session_state or st.session_state.all_stock_data.empty:
        st.warning("Please go to the 'MPT Analysis' page to select stocks and a date range.")
        return

    all_stock_data = st.session_state.all_stock_data

    st.subheader("1. Historical Data Retrieval")
    st.markdown("The first step is to retrieve historical stock data from Yahoo Finance for the selected stocks and date range.")
    st.code("all_stock_data = get_stock_data(selected_stocks, start_date, end_date)")
    st.dataframe(all_stock_data.head())

    st.subheader("2. Expected Annual Returns")
    st.markdown("The expected return of a portfolio is the weighted average of the expected returns of the individual assets in the portfolio.")
    st.latex(r'E(R_p) = \sum_{i=1}^{n} w_i E(R_i)')
    st.code("""daily_returns = all_stock_data.pct_change().dropna()
annual_returns = daily_returns.mean() * 252""")
    daily_returns = all_stock_data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    st.dataframe(annual_returns.to_frame(name="Annualized Expected Return"))

    st.subheader("3. Covariance Matrix")
    st.markdown("The covariance matrix measures the degree to which returns on two assets move in tandem.")
    st.latex(r'\sigma_{ij} = \text{Cov}(R_i, R_j)')
    st.code("annual_cov_matrix = daily_returns.cov() * 252")
    annual_cov_matrix = daily_returns.cov() * 252
    st.dataframe(annual_cov_matrix)

    st.subheader("4. Portfolio Return & Volatility")
    st.markdown("The return and volatility of a portfolio are calculated as follows:")
    st.latex(r'E(R_p) = w^T R')
    st.latex(r'\sigma_p^2 = w^T \Sigma w')
    st.code("""def get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate):
    returns = np.sum(weights * annual_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
    sharpe = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe""")

    st.subheader("5. Sharpe Ratio")
    st.markdown("The Sharpe Ratio measures the risk-adjusted return of a portfolio.")
    st.latex(r'\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}')
    st.code("sharpe = (returns - risk_free_rate) / volatility")

    st.subheader("6. Optimization")
    st.markdown("We use `scipy.optimize.minimize` to find the portfolio with the maximum Sharpe ratio.")
    st.code("""def neg_sharpe_ratio(weights, annual_returns, annual_cov_matrix, risk_free_rate):
    return -get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate)[2]

args = (annual_returns, annual_cov_matrix, risk_free_rate)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))
initial_weights = num_assets * [1. / num_assets]

max_sharpe_portfolio = minimize(neg_sharpe_ratio, initial_weights, args=args,
                                method='SLSQP', bounds=bounds, constraints=constraints)""")

def custom_mpt_page():
    st.header("Custom MPT Analysis")

    st.markdown("Enter your own ticker symbols and a date range to perform a complete MPT analysis.")

    # --- User Inputs ---
    tickers_input = st.text_input("Enter ticker symbols (comma-separated)", "AAPL,GOOG,MSFT")
    start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('today'))
    risk_free_rate_percent = st.slider("Risk-Free Rate (%)", min_value=2.0, max_value=7.0, value=4.0, step=0.1)
    risk_free_rate = risk_free_rate_percent / 100.0

    if st.button("Run Analysis"):
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

        if not tickers:
            st.warning("Please enter at least one ticker symbol.")
            return

        # --- Data Fetching ---
        all_stock_data = get_stock_data(tickers, start_date, end_date)

        if all_stock_data.empty:
            st.warning("Could not retrieve data for the specified tickers and date range.")
            return

        # --- MPT Analysis ---
        st.subheader("Stock History")
        st.dataframe(all_stock_data)

        st.subheader("Stock History Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        all_stock_data.plot(ax=ax)
        ax.set_title("Historical Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend(title="Stocks")
        ax.grid(True)
        st.pyplot(fig)

        daily_returns = all_stock_data.pct_change().dropna()
        annual_returns = daily_returns.mean() * 252
        annual_cov_matrix = daily_returns.cov() * 252
        num_assets = len(all_stock_data.columns)

        st.subheader("Annualized Returns")
        st.dataframe(annual_returns.to_frame(name="Annualized Expected Return"))

        st.subheader("Covariance Matrix")
        st.dataframe(annual_cov_matrix)

        # --- MPT Calculations ---
        st.subheader("Optimal Portfolio")

        def get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate):
            returns = np.sum(weights * annual_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
            sharpe = (returns - risk_free_rate) / volatility
            return returns, volatility, sharpe

        def neg_sharpe_ratio(weights, annual_returns, annual_cov_matrix, risk_free_rate):
            return -get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate)[2]

        def get_portfolio_volatility(weights, annual_cov_matrix):
            return get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate)[1]

        args = (annual_returns, annual_cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]

        max_sharpe_portfolio = minimize(neg_sharpe_ratio, initial_weights, args=args,
                                        method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = max_sharpe_portfolio.x
        optimal_return, optimal_volatility, optimal_sharpe = get_portfolio_performance(
            optimal_weights, annual_returns, annual_cov_matrix, risk_free_rate)

        st.write(f"**Optimal Portfolio (Max Sharpe Ratio):**")
        st.write(f"Expected Annual Return: {optimal_return:.2%}")
        st.write(f"Annual Volatility: {optimal_volatility:.2%}")
        st.write(f"Sharpe Ratio: {optimal_sharpe:.2f}")

        st.subheader("Optimal Portfolio Weights:")
        optimal_weights_df = pd.DataFrame({
            'Asset': all_stock_data.columns,
            'Weight': optimal_weights
        }).set_index('Asset')
        
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(optimal_weights_df.style.format({'Weight': '{:.2%}'}))

        with col2:
            fig, ax = plt.subplots()
            ax.pie(optimal_weights, labels=optimal_weights_df.index, autopct="%1.1f%%", startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

        # --- Efficient Frontier Envelope ---
        st.subheader("Efficient Frontier")
        frontier_returns = np.linspace(annual_returns.min(), annual_returns.max(), 100)
        frontier_volatilities = []

        for ret in frontier_returns:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},                                   {'type': 'eq', 'fun': lambda x: get_portfolio_performance(x, annual_returns, annual_cov_matrix, risk_free_rate)[0] - ret})
            result = minimize(get_portfolio_volatility, initial_weights, args=(annual_cov_matrix,),
                              method='SLSQP', bounds=bounds, constraints=constraints)
            frontier_volatilities.append(result['fun'])

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frontier_volatilities, frontier_returns, "b-", label="Efficient Frontier")
        ax.scatter(optimal_volatility, optimal_return, color="red", marker="*", s=300,
                   label="Optimal Portfolio (Max Sharpe Ratio)", edgecolors="black")
        cal_x = [0, optimal_volatility]
        cal_y = [risk_free_rate, optimal_return]
        ax.plot(cal_x, cal_y, color="blue", linestyle=":", label="Capital Allocation Line (CAL)")
        ax.set_title("Efficient Frontier with Optimal Portfolio")
        ax.set_xlabel("Annual Volatility (Standard Deviation)")
        ax.set_ylabel("Annual Expected Return")
        ax.grid(True)
        ax.set_ylim(bottom=0) # Set y-axis to start from 0
        ax.legend()
        st.pyplot(fig)

def mpt_analysis_page():
    st.header("Modern Portfolio Theory Analysis")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")

    # Date Range Selection
    today = date.today()
    default_start_date = today - timedelta(days=365) # 1 year ago

    start_date = st.sidebar.date_input("Start Date", value=default_start_date)
    end_date = st.sidebar.date_input("End Date", value=today)

    if start_date >= end_date:
        st.sidebar.error("Error: End Date must be after Start Date.")
        st.stop() # Stop execution if dates are invalid

    # List of 10 common stock tickers
    DEFAULT_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "JPM", "V", "PG", "KO"
    ]

    # User selects stocks
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to view:",
        options=DEFAULT_STOCKS,
        default=DEFAULT_STOCKS[:3] # Select a few by default
    )

    # Risk-Free Rate Selection
    risk_free_rate_percent = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.0, # Default 2%
        step=0.1
    )
    risk_free_rate = risk_free_rate_percent / 100.0

    # Optimize button
    optimize_button = st.sidebar.button("Optimize Portfolio")

    # --- Data Fetching ---
    all_stock_data = get_stock_data(selected_stocks, start_date, end_date)
    st.session_state.all_stock_data = all_stock_data # Store in session state

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Data & Plotting", "Returns & Covariance", "Efficient Frontier"])

    with tab1:
        st.header("Stock Data & Plotting")
        if all_stock_data.empty:
            st.warning("Please select stocks and a date range.")
        else:
            st.subheader("Historical Closing Prices (All Selected Stocks)")
            st.dataframe(all_stock_data)

            st.subheader("Closing Prices of Selected Stocks (Chart)")
            fig, ax = plt.subplots(figsize=(10, 6))
            all_stock_data.plot(ax=ax)
            ax.set_title("Historical Closing Prices")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend(title="Stocks")
            ax.grid(True)
            st.pyplot(fig)

    with tab2:
        st.header("Returns & Covariance")
        if all_stock_data.empty:
            st.warning("Please select stocks and a date range.")
        else:
            st.subheader("Daily Returns")
            daily_returns = all_stock_data.pct_change().dropna()
            st.dataframe(daily_returns)

            st.subheader("Annualized Expected Returns")
            annual_returns = daily_returns.mean() * 252
            st.dataframe(annual_returns.to_frame(name="Annualized Expected Return"))

            st.subheader("Annualized Covariance Matrix")
            annual_cov_matrix = daily_returns.cov() * 252
            st.dataframe(annual_cov_matrix)

    with tab3:
        st.header("Efficient Frontier and Optimal Portfolio")
        if all_stock_data.empty or len(all_stock_data.columns) < 2:
            st.warning("Please select at least two stocks and a date range.")
        else:
            if optimize_button:
                # --- MPT Calculations ---
                st.subheader("Portfolio Optimization Results")

                daily_returns = all_stock_data.pct_change().dropna()
                annual_returns = daily_returns.mean() * 252
                annual_cov_matrix = daily_returns.cov() * 252
                num_assets = len(all_stock_data.columns)

                # --- Helper functions for optimization ---
                def get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate):
                    returns = np.sum(weights * annual_returns)
                    volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
                    sharpe = (returns - risk_free_rate) / volatility
                    return returns, volatility, sharpe

                def neg_sharpe_ratio(weights, annual_returns, annual_cov_matrix, risk_free_rate):
                    return -get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate)[2]

                def get_portfolio_volatility(weights, annual_cov_matrix):
                    return get_portfolio_performance(weights, annual_returns, annual_cov_matrix, risk_free_rate)[1]

                # --- Max Sharpe Ratio Portfolio ---
                args = (annual_returns, annual_cov_matrix, risk_free_rate)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for asset in range(num_assets))
                initial_weights = num_assets * [1. / num_assets]

                max_sharpe_portfolio = minimize(neg_sharpe_ratio, initial_weights, args=args,
                                                method='SLSQP', bounds=bounds, constraints=constraints)

                optimal_weights = max_sharpe_portfolio.x
                optimal_return, optimal_volatility, optimal_sharpe = get_portfolio_performance(
                    optimal_weights, annual_returns, annual_cov_matrix, risk_free_rate)

                st.write(f"**Optimal Portfolio (Max Sharpe Ratio):**")
                st.write(f"Risk-Free Rate: {risk_free_rate:.2%}")
                st.write(f"Expected Annual Return: {optimal_return:.2%}")
                st.write(f"Annual Volatility: {optimal_volatility:.2%}")
                st.write(f"Sharpe Ratio: {optimal_sharpe:.2f}")

                st.subheader("Optimal Portfolio Weights:")
                optimal_weights_df = pd.DataFrame({
                    'Asset': all_stock_data.columns,
                    'Weight': optimal_weights
                }).set_index('Asset')
                
                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(optimal_weights_df.style.format({'Weight': '{:.2%}'}))

                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(optimal_weights, labels=optimal_weights_df.index, autopct="%1.1f%%", startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)

                # --- Efficient Frontier Envelope ---
                st.subheader("Efficient Frontier")
                frontier_returns = np.linspace(annual_returns.min(), annual_returns.max(), 100)
                frontier_volatilities = []

                for ret in frontier_returns:
                    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},                                   {'type': 'eq', 'fun': lambda x: get_portfolio_performance(x, annual_returns, annual_cov_matrix, risk_free_rate)[0] - ret})
                    result = minimize(get_portfolio_volatility, initial_weights, args=(annual_cov_matrix,),
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                    frontier_volatilities.append(result['fun'])

                # --- Plotting ---
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(frontier_volatilities, frontier_returns, "b-", label="Efficient Frontier")
                ax.scatter(optimal_volatility, optimal_return, color="red", marker="*", s=300,
                           label="Optimal Portfolio (Max Sharpe Ratio)", edgecolors="black")
                cal_x = [0, optimal_volatility]
                cal_y = [risk_free_rate, optimal_return]
                ax.plot(cal_x, cal_y, color="blue", linestyle=":", label="Capital Allocation Line (CAL)")
                ax.set_title("Efficient Frontier with Optimal Portfolio")
                ax.set_xlabel("Annual Volatility (Standard Deviation)")
                ax.set_ylabel("Annual Expected Return")
                ax.grid(True)
                ax.set_ylim(bottom=0) # Set y-axis to start from 0
                ax.legend()
                st.pyplot(fig)

            else:
                st.info("Click the 'Optimize Portfolio' button in the sidebar.")

def code_page():
    st.header("View and Download Code")
    st.write("Here you can view the source code of this application.")

    # Read the content of the current app.py file
    try:
        with open("gemini_cli_app.py", "r") as f:
            code_content = f.read()
        st.code(code_content, language="python")

        st.download_button(
            label="Download app.py",
            data=code_content,
            file_name="mpt_app.py",
            mime="text/plain"
        )
    except FileNotFoundError:
        st.error("Could not find 'app.py' to display its content.")


# --- Main App Logic ---

st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ("Introduction to MPT", "Mathematical Foundation", "MPT Analysis", "Step-by-Step Guide", "Custom MPT Analysis", "Code")
)

# --- Function Definitions ---
@st.cache_data
def get_stock_data(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    all_data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end)
            if not data.empty:
                all_data[ticker] = data["Close"]
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    return all_data


# --- Page Dispatcher ---
if page_selection == "Introduction to MPT":
    intro_mpt_page()
elif page_selection == "MPT Analysis":
    mpt_analysis_page()
elif page_selection == "Mathematical Foundation":
    math_foundation_page()
elif page_selection == "Step-by-Step Guide":
    step_by_step_page()
elif page_selection == "Custom MPT Analysis":
    custom_mpt_page()
elif page_selection == "Code":
    code_page()
