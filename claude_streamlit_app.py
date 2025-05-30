import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Modern Portfolio Theory App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .equation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PortfolioOptimizer:
    def __init__(self, tickers, start_date=None, end_date=None):
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.risk_free_rate = 0.02
        
    def fetch_data(self):
        try:
            self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
            if len(self.tickers) == 1:
                self.data = self.data.to_frame()
                self.data.columns = self.tickers
            self.data = self.data.dropna(axis=1, how='all')
            self.data = self.data.dropna()
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False
    
    def calculate_returns(self):
        if self.data is None:
            return False
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        return True
    
    def portfolio_performance(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights):
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        return self.portfolio_performance(weights)[1]
    
    def find_optimal_portfolio(self):
        num_assets = len(self.mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
        result = minimize(self.negative_sharpe_ratio, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            performance = self.portfolio_performance(optimal_weights)
            return {
                'weights': optimal_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        return None
    
    def find_min_variance_portfolio(self):
        num_assets = len(self.mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
        result = minimize(self.portfolio_volatility, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            min_var_weights = result.x
            performance = self.portfolio_performance(min_var_weights)
            return {
                'weights': min_var_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        return None
    
    def generate_efficient_frontier(self, num_portfolios=100):
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        for target in target_returns:
            try:
                portfolio = self.find_efficient_portfolio(target)
                if portfolio:
                    efficient_portfolios.append(portfolio)
            except:
                continue
        return efficient_portfolios
    
    def find_efficient_portfolio(self, target_return):
        num_assets = len(self.mean_returns)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
        result = minimize(self.portfolio_volatility, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            performance = self.portfolio_performance(weights)
            return {
                'weights': weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        return None

# Sidebar navigation
def main():
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📚 Introduction to MPT", "🧮 Mathematical Foundation", "💻 Code Implementation", 
         "📊 Portfolio Optimizer", "📈 Browse Stock History"]
    )
    
    if page == "📚 Introduction to MPT":
        show_introduction()
    elif page == "🧮 Mathematical Foundation":
        show_mathematical_foundation()
    elif page == "💻 Code Implementation":
        show_code_implementation()
    elif page == "📊 Portfolio Optimizer":
        show_portfolio_optimizer()
    elif page == "📈 Browse Stock History":
        show_stock_history()

def show_introduction():
    st.markdown('<h1 class="main-header">📚 Introduction to Modern Portfolio Theory</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## What is Modern Portfolio Theory?
        
        Modern Portfolio Theory (MPT), developed by **Harry Markowitz** in 1952, is a mathematical framework 
        for constructing portfolios that optimize the trade-off between expected return and risk.
        
        ### Key Concepts:
        
        **🎯 Diversification**
        - Don't put all your eggs in one basket
        - Combining uncorrelated assets reduces overall portfolio risk
        - The whole can be less risky than the sum of its parts
        
        **⚖️ Risk-Return Trade-off**
        - Higher expected returns come with higher risk
        - Investors must balance their desire for returns with their tolerance for risk
        - There's no free lunch in investing
        
        **📈 Efficient Frontier**
        - The set of optimal portfolios offering the highest expected return for each level of risk
        - Portfolios below the frontier are sub-optimal
        - Rational investors should only consider portfolios on the frontier
        """)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🏆 Nobel Prize Achievement
        Harry Markowitz won the **1990 Nobel Prize in Economics** for developing Modern Portfolio Theory, 
        which revolutionized investment management and became the foundation of modern finance.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Create a simple risk-return visualization
        fig = go.Figure()
        
        # Sample data for illustration
        risk = [0.1, 0.15, 0.2, 0.25, 0.3]
        return_vals = [0.05, 0.08, 0.12, 0.15, 0.18]
        
        fig.add_trace(go.Scatter(
            x=risk, y=return_vals,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Risk-Return Trade-off",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ## Core Principles of MPT
    
    ### 1. 📊 Quantitative Approach
    MPT uses mathematical models to make investment decisions based on:
    - Historical return data
    - Statistical measures of risk (variance/standard deviation)
    - Correlation between assets
    
    ### 2. 🎲 Risk Management
    - Risk is measured as the volatility of returns (standard deviation)
    - Diversification reduces risk without necessarily reducing expected returns
    - Correlation between assets is crucial for effective diversification
    
    ### 3. 🎯 Optimization
    - Find the portfolio that maximizes return for a given level of risk
    - Or minimize risk for a given level of expected return
    - Use mathematical optimization techniques
    
    ### 4. 📈 Rational Investor Assumptions
    - Investors are rational and risk-averse
    - Investors prefer higher returns to lower returns
    - Investors prefer lower risk to higher risk for the same return
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🚀 Benefits
        - **Scientific approach** to portfolio construction
        - **Quantifies** risk and return
        - **Optimizes** diversification
        - **Foundation** for modern finance
        """)
        
    with col2:
        st.markdown("""
        ### ⚠️ Limitations
        - Assumes **normal distribution** of returns
        - Based on **historical data**
        - Ignores **transaction costs**
        - Requires **stable correlations**
        """)
        
    with col3:
        st.markdown("""
        ### 🔄 Modern Applications
        - **Robo-advisors** use MPT algorithms
        - **Pension funds** for asset allocation
        - **Mutual funds** portfolio construction
        - **Risk management** systems
        """)

def show_mathematical_foundation():
    st.markdown('<h1 class="main-header">🧮 Mathematical Foundation of MPT</h1>', unsafe_allow_html=True)
    
    st.markdown("## Core Mathematical Concepts")
    
    # Expected Return
    st.markdown('<div class="sub-header">📊 Expected Portfolio Return</div>', unsafe_allow_html=True)
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.latex(r'''E(R_p) = \sum_{i=1}^{n} w_i \cdot E(R_i)''')
    st.markdown("""
    Where:
    - $E(R_p)$ = Expected return of the portfolio
    - $w_i$ = Weight of asset $i$ in the portfolio  
    - $E(R_i)$ = Expected return of asset $i$
    - $\sum_{i=1}^{n} w_i = 1$ (weights sum to 100%)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Portfolio Variance
    st.markdown('<div class="sub-header">📈 Portfolio Risk (Variance)</div>', unsafe_allow_html=True)
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.latex(r'''\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i \cdot w_j \cdot \sigma_{ij}''')
    st.markdown("""
    This can be expanded as:
    """)
    st.latex(r'''\sigma_p^2 = \sum_{i=1}^{n} w_i^2 \cdot \sigma_i^2 + \sum_{i=1}^{n} \sum_{\substack{j=1 \\ j \neq i}}^{n} w_i \cdot w_j \cdot \sigma_{ij}''')
    st.markdown("""
    Where:
    - $\sigma_p^2$ = Portfolio variance (risk)
    - $\sigma_{ij}$ = Covariance between assets $i$ and $j$
    - $\sigma_i^2$ = Variance of asset $i$
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Covariance and Correlation
    st.markdown('<div class="sub-header">🔗 Covariance and Correlation</div>', unsafe_allow_html=True)
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.latex(r'''\sigma_{ij} = \rho_{ij} \cdot \sigma_i \cdot \sigma_j''')
    st.markdown("""
    Where:
    - $ρ_{ij}$ = Correlation coefficient between assets $i$ and $j$ (-1 ≤ ρ ≤ 1)
    - $\sigma_i, \sigma_j$ = Standard deviations of assets $i$ and $j$
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optimization Problem
    st.markdown('<div class="sub-header">🎯 Optimization Problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.markdown("**Minimize:**")
    st.latex(r'''\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i \cdot w_j \cdot \sigma_{ij}''')
    st.markdown("**Subject to:**")
    st.latex(r'''\sum_{i=1}^{n} w_i = 1 \text{ (budget constraint)}''')
    st.latex(r'''\sum_{i=1}^{n} w_i \cdot E(R_i) = \mu \text{ (target return constraint)}''')
    st.latex(r'''w_i \geq 0 \text{ (no short selling)}''')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sharpe Ratio
    st.markdown('<div class="sub-header">📊 Sharpe Ratio</div>', unsafe_allow_html=True)
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.latex(r'''S = \frac{E(R_p) - r_f}{\sigma_p}''')
    st.markdown("""
    Where:
    - $S$ = Sharpe ratio (risk-adjusted return)
    - $r_f$ = Risk-free rate
    - Higher Sharpe ratio indicates better risk-adjusted performance
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Matrix Notation
    st.markdown('<div class="sub-header">🔢 Matrix Notation</div>', unsafe_allow_html=True)
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.latex(r'''\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}''')
    st.markdown("""
    Where:
    - $\mathbf{w}$ = Vector of portfolio weights
    - $\Sigma$ = Covariance matrix of asset returns
    - $\mathbf{w}^T$ = Transpose of the weight vector
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive Example
    st.markdown("## 🧪 Interactive Example: Two-Asset Portfolio")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Portfolio Parameters")
        w1 = st.slider("Weight of Asset 1", 0.0, 1.0, 0.6, 0.01)
        w2 = 1 - w1
        st.write(f"Weight of Asset 2: {w2:.2f}")
        
        r1 = st.slider("Expected Return Asset 1 (%)", 0.0, 20.0, 10.0, 0.5) / 100
        r2 = st.slider("Expected Return Asset 2 (%)", 0.0, 20.0, 15.0, 0.5) / 100
        
        sigma1 = st.slider("Volatility Asset 1 (%)", 1.0, 30.0, 15.0, 0.5) / 100
        sigma2 = st.slider("Volatility Asset 2 (%)", 1.0, 30.0, 25.0, 0.5) / 100
        
        correlation = st.slider("Correlation", -1.0, 1.0, 0.3, 0.01)
    
    with col2:
        # Calculate portfolio metrics
        portfolio_return = w1 * r1 + w2 * r2
        covariance = correlation * sigma1 * sigma2
        portfolio_variance = (w1**2 * sigma1**2) + (w2**2 * sigma2**2) + (2 * w1 * w2 * covariance)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Display results
        st.markdown("### Portfolio Results")
        st.metric("Portfolio Return", f"{portfolio_return*100:.2f}%")
        st.metric("Portfolio Volatility", f"{portfolio_volatility*100:.2f}%")
        st.metric("Diversification Benefit", f"{((w1*sigma1 + w2*sigma2) - portfolio_volatility)*100:.2f}%")
        
        # Show the impact of correlation
        st.markdown("### Impact of Correlation")
        correlations = np.linspace(-1, 1, 100)
        volatilities = []
        
        for corr in correlations:
            cov = corr * sigma1 * sigma2
            var = (w1**2 * sigma1**2) + (w2**2 * sigma2**2) + (2 * w1 * w2 * cov)
            volatilities.append(np.sqrt(var))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=correlations, y=np.array(volatilities)*100,
            mode='lines',
            name='Portfolio Volatility',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.add_vline(x=correlation, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Portfolio Volatility vs Correlation",
            xaxis_title="Correlation",
            yaxis_title="Portfolio Volatility (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_code_implementation():
    st.markdown('<h1 class="main-header">💻 Code Implementation</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📥 Download Complete Application
    
    You can download the complete Streamlit application code below:
    """)
    
    # Get the current script content for download
    with open(__file__, 'r') as file:
        app_code = file.read()
    
    # Create download button
    st.download_button(
        label="📥 Download Complete App Code",
        data=app_code,
        file_name="mpt_streamlit_app.py",
        mime="text/plain"
    )
    
    st.markdown("---")
    
    st.markdown("""
    ## 🛠️ Installation Requirements
    
    To run this application, you need to install the following packages:
    """)
    
    requirements = """
streamlit
numpy
pandas
yfinance
matplotlib
plotly
scipy
seaborn
"""
    
    st.code(requirements, language="text")
    
    st.markdown("""
    Install all requirements using:
    ```bash
    pip install streamlit numpy pandas yfinance matplotlib plotly scipy seaborn
    ```
    
    Run the application with:
    ```bash
    streamlit run mpt_streamlit_app.py
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🔍 Code Structure Overview
    
    The application is structured into several key components:
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Optimizer Class", "Data Fetching", "Optimization Logic", "Visualization"])
    
    with tab1:
        st.markdown("### PortfolioOptimizer Class")
        st.code("""
class PortfolioOptimizer:
    def __init__(self, tickers, start_date=None, end_date=None):
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.risk_free_rate = 0.02
        """, language="python")
    
    with tab2:
        st.markdown("### Data Fetching with yfinance")
        st.code("""
def fetch_data(self):
    try:
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        if len(self.tickers) == 1:
            self.data = self.data.to_frame()
            self.data.columns = self.tickers
        self.data = self.data.dropna(axis=1, how='all')
        self.data = self.data.dropna()
        return True
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return False
        """, language="python")
    
    with tab3:
        st.markdown("### Portfolio Optimization")
        st.code("""
def find_optimal_portfolio(self):
    num_assets = len(self.mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1/num_assets] * num_assets)
    
    result = minimize(self.negative_sharpe_ratio, initial_guess,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        performance = self.portfolio_performance(optimal_weights)
        return {
            'weights': optimal_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    return None
        """, language="python")
    
    with tab4:
        st.markdown("### Plotly Visualizations")
        st.code("""
# Create interactive efficient frontier plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=volatilities, y=returns,
    mode='markers',
    marker=dict(
        color=sharpe_ratios,
        colorscale='Viridis',
        size=8,
        colorbar=dict(title="Sharpe Ratio")
    ),
    name='Efficient Frontier'
))

fig.update_layout(
    title='Efficient Frontier',
    xaxis_title='Volatility (Risk)',
    yaxis_title='Expected Return',
    hovermode='closest'
)
        """, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Key Features Implemented
    
    - **Interactive Streamlit Interface**: Multi-page navigation with sidebar
    - **Real-time Data Fetching**: Using yfinance for current market data
    - **Mathematical Optimization**: Scipy minimize for finding optimal portfolios
    - **Interactive Visualizations**: Plotly charts for better user experience
    - **Portfolio Performance Tracking**: Historical portfolio growth simulation
    - **Stock Analysis Tools**: Individual stock history and statistics
    
    ## 🔧 Customization Options
    
    The code is designed to be easily customizable:
    - Add new optimization constraints
    - Modify risk-free rate assumptions
    - Include transaction costs
    - Add additional performance metrics
    - Extend visualization capabilities
    """)

def show_portfolio_optimizer():
    st.markdown('<h1 class="main-header">📊 Portfolio Optimizer</h1>', unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.markdown("## 🎛️ Portfolio Settings")
    
    # Stock selection
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    stock_input = st.sidebar.text_area(
        "Enter stock tickers (one per line):",
        value='\n'.join(default_stocks),
        height=150
    )
    
    tickers = [ticker.strip().upper() for ticker in stock_input.split('\n') if ticker.strip()]
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    risk_free_rate = st.sidebar.slider("Risk-free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    
    if st.sidebar.button("🚀 Optimize Portfolio", type="primary"):
        if len(tickers) < 2:
            st.error("Please select at least 2 stocks for portfolio optimization.")
            return
        
        with st.spinner("Fetching data and optimizing portfolio..."):
            optimizer = PortfolioOptimizer(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            optimizer.risk_free_rate = risk_free_rate
            
            if optimizer.fetch_data() and optimizer.calculate_returns():
                # Store in session state
                st.session_state.optimizer = optimizer
                st.session_state.optimization_complete = True
                st.success("✅ Portfolio optimization completed!")
            else:
                st.error("❌ Failed to fetch data or optimize portfolio.")
                return
    
    if hasattr(st.session_state, 'optimization_complete') and st.session_state.optimization_complete:
        optimizer = st.session_state.optimizer
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Efficient Frontier", "⚖️ Optimal Weights", "📊 Performance Metrics", "📈 Portfolio Growth"])
        
        with tab1:
            st.markdown("### Efficient Frontier")
            
            # Generate efficient frontier
            efficient_portfolios = optimizer.generate_efficient_frontier(50)
            
            if efficient_portfolios:
                returns = [p['expected_return'] for p in efficient_portfolios]
                volatilities = [p['volatility'] for p in efficient_portfolios]
                sharpe_ratios = [p['sharpe_ratio'] for p in efficient_portfolios]
                
                # Get optimal and min variance portfolios
                optimal = optimizer.find_optimal_portfolio()
                min_var = optimizer.find_min_variance_portfolio()
                
                # Create efficient frontier plot
                fig = go.Figure()
                
                # Efficient frontier
                fig.add_trace(go.Scatter(
                    x=np.array(volatilities)*100, y=np.array(returns)*100,
                    mode='markers',
                    marker=dict(
                        color=sharpe_ratios,
                        colorscale='Viridis',
                        size=8,
                        colorbar=dict(title="Sharpe Ratio"),
                        line=dict(width=1, color='white')
                    ),
                    name='Efficient Frontier',
                    hovertemplate='<b>Risk:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<br><b>Sharpe:</b> %{marker.color:.3f}<extra></extra>'
                ))
                
                # Individual assets
                for i, ticker in enumerate(optimizer.mean_returns.index):
                    fig.add_trace(go.Scatter(
                        x=[np.sqrt(optimizer.cov_matrix.iloc[i, i])*100],
                        y=[optimizer.mean_returns.iloc[i]*100],
                        mode='markers',
                        marker=dict(symbol='star', size=15, color='red'),
                        name=ticker,
                        hovertemplate=f'<b>{ticker}</b><br><b>Risk:</b> %{{x:.2f}}%<br><b>Return:</b> %{{y:.2f}}%<extra></extra>'
                    ))
                
                # Optimal portfolio
                if optimal:
                    fig.add_trace(go.Scatter(
                        x=[optimal['volatility']*100],
                        y=[optimal['expected_return']*100],
                        mode='markers',
                        marker=dict(symbol='diamond', size=20, color='gold', line=dict(width=2, color='black')),
                        name=f'Optimal (Sharpe: {optimal["sharpe_ratio"]:.3f})',
                        hovertemplate='<b>Optimal Portfolio</b><br><b>Risk:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<extra></extra>'
                    ))
                
                # Min variance portfolio
                if min_var:
                    fig.add_trace(go.Scatter(
                        x=[min_var['volatility']*100],
                        y=[min_var['expected_return']*100],
                        mode='markers',
                        marker=dict(symbol='square', size=20, color='orange', line=dict(width=2, color='black')),
                        name=f'Min Variance (Vol: {min_var["volatility"]*100:.2f}%)',
                        hovertemplate='<b>Min Variance Portfolio</b><br><b>Risk:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='Efficient Frontier Analysis',
                    xaxis_title='Volatility (Risk) %',
                    yaxis_title='Expected Return %',
                    hovermode='closest',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate efficient frontier.")
        
        with tab2:
            st.markdown("### Optimal Portfolio Weights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Optimal portfolio
                optimal = optimizer.find_optimal_portfolio()
                if optimal:
                    st.markdown("#### 🏆 Maximum Sharpe Ratio Portfolio")
                    weights_df = pd.DataFrame({
                        'Asset': optimizer.mean_returns.index,
                        'Weight': optimal['weights'],
                        'Weight %': optimal['weights'] * 100
                    }).sort_values('Weight %', ascending=False)
                    
                    # Pie chart for optimal weights
                    fig_pie1 = px.pie(
                        weights_df, 
                        values='Weight %', 
                        names='Asset',
                        title='Optimal Portfolio Allocation'
                    )
                    st.plotly_chart(fig_pie1, use_container_width=True)
                    
                    st.dataframe(weights_df, use_container_width=True)
            
            with col2:
                # Min variance portfolio
                min_var = optimizer.find_min_variance_portfolio()
                if min_var:
                    st.markdown("#### 🛡️ Minimum Variance Portfolio")
                    min_var_weights_df = pd.DataFrame({
                        'Asset': optimizer.mean_returns.index,
                        'Weight': min_var['weights'],
                        'Weight %': min_var['weights'] * 100
                    }).sort_values('Weight %', ascending=False)
                    
                    # Pie chart for min variance weights
                    fig_pie2 = px.pie(
                        min_var_weights_df, 
                        values='Weight %', 
                        names='Asset',
                        title='Min Variance Portfolio Allocation'
                    )
                    st.plotly_chart(fig_pie2, use_container_width=True)
                    
                    st.dataframe(min_var_weights_df, use_container_width=True)
        
        with tab3:
            st.markdown("### Portfolio Performance Metrics")
            
            optimal = optimizer.find_optimal_portfolio()
            min_var = optimizer.find_min_variance_portfolio()
            
            if optimal and min_var:
                # Performance comparison
                performance_data = {
                    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                    'Optimal Portfolio': [f"{optimal['expected_return']*100:.2f}%", 
                                        f"{optimal['volatility']*100:.2f}%", 
                                        f"{optimal['sharpe_ratio']:.3f}"],
                    'Min Variance Portfolio': [f"{min_var['expected_return']*100:.2f}%", 
                                             f"{min_var['volatility']*100:.2f}%", 
                                             f"{min_var['sharpe_ratio']:.3f}"]
                }
                
                performance_df = pd.DataFrame(performance_data)
                st.table(performance_df)
                
                # Risk-Return metrics for individual assets
                st.markdown("#### Individual Asset Statistics")
                asset_stats = pd.DataFrame({
                    'Asset': optimizer.mean_returns.index,
                    'Expected Return': optimizer.mean_returns.values * 100,
                    'Volatility': np.sqrt(np.diag(optimizer.cov_matrix)) * 100,
                    'Sharpe Ratio': (optimizer.mean_returns.values - optimizer.risk_free_rate) / np.sqrt(np.diag(optimizer.cov_matrix))
                }).round(3)
                
                st.dataframe(asset_stats, use_container_width=True)
                
                # Correlation matrix
                st.markdown("#### Asset Correlation Matrix")
                corr_matrix = optimizer.returns.corr()
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                fig_heatmap.update_layout(title="Asset Correlation Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab4:
            st.markdown("### Portfolio Growth Simulation")
            
            optimal = optimizer.find_optimal_portfolio()
            if optimal:
                # Calculate portfolio returns
                portfolio_returns = (optimizer.returns * optimal['weights']).sum(axis=1)
                
                # Calculate cumulative returns
                cumulative_returns = (1 + portfolio_returns).cumprod()
                
                # Calculate individual asset cumulative returns for comparison
                individual_cumulative = (1 + optimizer.returns).cumprod()
                
                # Create growth chart
                fig_growth = go.Figure()
                
                # Portfolio growth
                fig_growth.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='Optimal Portfolio',
                    line=dict(color='gold', width=3)
                ))
                
                # Individual assets growth
                for ticker in optimizer.returns.columns:
                    fig_growth.add_trace(go.Scatter(
                        x=individual_cumulative.index,
                        y=individual_cumulative[ticker].values,
                        mode='lines',
                        name=ticker,
                        opacity=0.7
                    ))
                
                fig_growth.update_layout(
                    title='Portfolio Growth Comparison',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_growth, use_container_width=True)
                
                # Performance statistics
                col1, col2, col3, col4 = st.columns(4)
                
                total_return = cumulative_returns.iloc[-1] - 1
                annual_return = portfolio_returns.mean() * 252
                annual_vol = portfolio_returns.std() * np.sqrt(252)
                max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min()
                
                with col1:
                    st.metric("Total Return", f"{total_return*100:.2f}%")
                with col2:
                    st.metric("Annual Return", f"{annual_return*100:.2f}%")
                with col3:
                    st.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")

def show_stock_history():
    st.markdown('<h1 class="main-header">📈 Browse Stock History</h1>', unsafe_allow_html=True)
    
    # Stock selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL", help="e.g., AAPL, MSFT, GOOGL")
    
    with col2:
        period = st.selectbox("Time Period:", ["1y", "2y", "5y", "10y", "max"])
    
    if ticker:
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker.upper())
            hist_data = stock.history(period=period)
            info = stock.info
            
            if not hist_data.empty:
                # Display stock info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${hist_data['Close'].iloc[-1]:.2f}")
                with col2:
                    change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
                    change_pct = (change / hist_data['Close'].iloc[-2]) * 100
                    st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
                with col3:
                    st.metric("52-Week High", f"${hist_data['High'].max():.2f}")
                with col4:
                    st.metric("52-Week Low", f"${hist_data['Low'].min():.2f}")
                
                # Price chart
                fig_price = go.Figure()
                
                fig_price.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig_price.update_layout(
                    title=f'{ticker.upper()} Stock Price History',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure()
                
                fig_volume.add_trace(go.Bar(
                    x=hist_data.index,
                    y=hist_data['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                
                fig_volume.update_layout(
                    title=f'{ticker.upper()} Trading Volume',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    height=300
                )
                
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Returns analysis
                returns = hist_data['Close'].pct_change().dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Returns distribution
                    fig_hist = px.histogram(
                        returns*100, 
                        nbins=50,
                        title=f'{ticker.upper()} Daily Returns Distribution',
                        labels={'value': 'Daily Return (%)', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Statistics
                    st.markdown("#### 📊 Statistical Summary")
                    stats_data = {
                        'Metric': ['Mean Daily Return', 'Daily Volatility', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                        'Value': [
                            f"{returns.mean()*100:.3f}%",
                            f"{returns.std()*100:.3f}%",
                            f"{returns.mean()*252*100:.2f}%",
                            f"{returns.std()*np.sqrt(252)*100:.2f}%",
                            f"{(returns.mean()*252 - 0.02) / (returns.std()*np.sqrt(252)):.3f}"
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.table(stats_df)
                
                # Company information
                if info:
                    st.markdown("#### 🏢 Company Information")
                    company_info = {
                        'Company Name': info.get('longName', 'N/A'),
                        'Sector': info.get('sector', 'N/A'),
                        'Industry': info.get('industry', 'N/A'),
                        'Market Cap': f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else 'N/A',
                        'P/E Ratio': f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A'
                    }
                    
                    for key, value in company_info.items():
                        st.write(f"**{key}:** {value}")
                
                # Download data option
                csv = hist_data.to_csv()
                st.download_button(
                    label=f"📥 Download {ticker.upper()} Data",
                    data=csv,
                    file_name=f"{ticker.upper()}_historical_data.csv",
                    mime="text/csv"
                )
                
            else:
                st.error(f"No data found for ticker: {ticker.upper()}")
        
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    main()
