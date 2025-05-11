import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import base64
from io import BytesIO
import datetime
import seaborn as sns
import random

# Set page configuration
st.set_page_config(
    page_title="Modern Portfolio Theory Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3498db;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #3498db;
    }
    .formula {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-style: italic;
    }
    .caption {
        font-size: 0.9rem;
        font-style: italic;
        color: #7f8c8d;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Markowitz_frontier.svg/440px-Markowitz_frontier.svg.png", 
             # caption="Efficient Frontier Concept")
    st.title("Modern Portfolio Theory\nAn Illustration Using Dow Jones Industrials")
    st.markdown("---")   
    st.markdown("### Navigation")
    page = st.radio("", ["Introduction", "Mathematical Foundations", "Interactive Portfolio Optimizer","Stocks Price History", "Code Implementation", "About"])   
    st.markdown("---")
    # st.markdown("Select parameters and\n click checkbox below")	
    # st.markdown("### Parameters")
    if page == "Interactive Portfolio Optimizer":
      st.markdown("### Parameters")
      st.markdown("Select parameters and\n click checkbox below")	
      min_date = datetime.date(2015, 1, 1)
      max_date = datetime.date(2025, 4, 9)
      d1 = st.date_input("Stocks History Start Date:", value=datetime.date(2018, 1, 1), min_value=min_date, max_value=max_date)
      d2 = st.date_input("Stocks History End Date:", value=datetime.date(2023, 1, 1), min_value=min_date, max_value=max_date)
      #num_assets = st.slider("Number of Assets", min_value=3, max_value=12, value=10, step=1)
      risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100
      num_portfolios = st.slider("Number of Random Portfolios", min_value=1000, max_value=10000, value=5000, step=1000)
      
      prices = pd.read_csv("dowjones.csv", index_col=0, nrows=2)
      prices.index = pd.to_datetime(prices.index)
      stocks_list = prices.columns
      
      symbols = st.multiselect("Choose at least 3 DOW Stocks:", stocks_list, stocks_list)

      rnd = 0
      str1 = ""
      if len(symbols) < 3:
          symbols=random.sample(list(stocks_list), 3)
          rnd = 1
          str1 = "3 random DOW stocks " + str(symbols) + " have been selected"
      num_assets = len(symbols)
      if rnd == 0:
        st.write(str(num_assets) + " DOW stock symbols " + str(symbols) + " selected for analysis")
      else:
        st.write(str1)
		  
      opt = st.checkbox("Compute optimal porfolio\n(may roughly take a maximum of 3 min for all 30 stocks)",key="my_checkbox")
          
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("[Markowitz's Original Paper](https://www.jstor.org/stable/2975974)")
    st.markdown("[Wikipedia: Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)")
    st.markdown("[Investopedia: MPT](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)")
    
def view_stock_history(stockfilename):
  # Load the CSV data
  df = pd.read_csv(stockfilename)
  # Convert 'Date' column to datetime objects
  df['Date'] = pd.to_datetime(df['Date'])
  
  df = df.set_index('Date')  
  min_date = datetime.date(2015, 1, 1)
  max_date = datetime.date.today()
  d1 = st.date_input("Stocks History Start Date:", value=datetime.date(2015, 1, 1), min_value=min_date, max_value=max_date)
  d2 = st.date_input("Stocks History End Date:", value=datetime.date.today(), min_value=min_date, max_value=max_date)
  start_date = pd.to_datetime(d1)   #'2018-01-01'
  end_date = pd.to_datetime(d2)     #'2023-01-01'
  df = df[(df.index >= start_date) & (df.index <= end_date)]



  # Get the list of company names (excluding the 'Date' column)
  company_names = df.columns.tolist()

  # Create a multiselect widget for company selection
  selected_companies = st.multiselect("Select companies to display:", company_names, default=company_names[:3])
  #selected_companies = st.multiselect("Select companies to display:", company_names, default=random.sample(company_names, 3))
  # Filter the DataFrame based on the selected companies
  if selected_companies:
    filtered_df = df[selected_companies]

    # Create the Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='x', labelrotation=45)

    # Plot the close price history for each selected company
    for company in filtered_df.columns:
        ax.plot(filtered_df.index, filtered_df[company], label=company, lw=0.5)

    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
	  
    ax.set_title("Daily Stock Close Price History")
    plt.title("Daily Stock Close Price History", fontsize=6)  # Set title font size
    plt.xlabel("Date", fontsize=6)  # Set x-axis label font size
    plt.ylabel("Close Price", fontsize=6)  # Set y-axis label font size
    plt.xticks(fontsize=6)  # Set x-axis tick labels font size
    plt.yticks(fontsize=6)  # Set y-axis tick labels font size	  
	  
    ax.legend(fontsize=6)
    ax.grid(True)
    #plt.rc('font', size=20)
    fig.tight_layout()

    # Display the Matplotlib chart in Streamlit
    st.pyplot(fig)
    st.write("Stocks price history")
    st.dataframe(filtered_df)
    mean_row = pd.DataFrame(filtered_df.pct_change().dropna().mean().values.reshape(1, -1),
                            columns=filtered_df.pct_change().dropna().mean().index,
                            index=['Annual Pct Return'])

    std_row = pd.DataFrame(filtered_df.pct_change().dropna().std().values.reshape(1, -1),
                            columns=filtered_df.pct_change().dropna().std().index,
                            index=['Annual Std'])

    st.write(pd.concat([100*252*mean_row, np.sqrt(100*252*std_row)]))                     
  else:
    st.warning("Please select at least one company to display.")
    
    
def generate_sample_data(num_assets=10, num_observations=252):
    """Generate sample return data for assets."""
    asset_names = [f"Asset_{i+1}" for i in range(num_assets)]
    
    stocks = symbols[0:num_assets]
    start_date = pd.to_datetime(d1)   #'2018-01-01'
    end_date = pd.to_datetime(d2)     #'2023-01-01'
    prices = pd.read_csv("dowjones.csv", index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
    prices = prices[stocks]
    
    asset_names = [prices.columns[i] for i in range(num_assets)]
    
	# Calculate daily returns
    returns1 = prices.pct_change().dropna()
    corr_matrix1 = returns1.corr()   
    
    # Generate random returns with different means and standard deviations
    annual_returns = np.random.uniform(0.05, 0.15, num_assets)
    daily_returns = annual_returns / num_observations
    
    annual_volatilities = np.random.uniform(0.10, 0.30, num_assets)
    daily_volatilities = annual_volatilities / np.sqrt(num_observations)
    
    # Generate correlation matrix
    corr_matrix = np.random.uniform(0.1, 0.7, (num_assets, num_assets))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Ensure the matrix is positive-semidefinite
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        corr_matrix += np.eye(num_assets) * (abs(min_eig) + 1e-8)
    
    # Normalize to ensure it's a proper correlation matrix
    D = np.diag(1.0 / np.sqrt(np.diag(corr_matrix)))
    corr_matrix = D @ corr_matrix @ D
    
    # Generate returns using multivariate normal distribution
    cov_matrix = np.outer(daily_volatilities, daily_volatilities) * corr_matrix
    returns = np.random.multivariate_normal(
        mean=daily_returns,
        cov=cov_matrix,
        size=num_observations
    )
    
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    #return returns_df, corr_matrix
    return returns1, corr_matrix1, prices   

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio performance metrics."""
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """Calculate the negative Sharpe ratio (for minimization)."""
    p_return, p_volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    
    return -sharpe_ratio

def min_portfolio_volatility(weights, mean_returns, cov_matrix):
    """Calculate portfolio volatility (for minimization)."""
    return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]

def optimize_portfolio(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.01):
    """Optimize portfolio for maximum Sharpe ratio and minimum volatility."""
    num_assets = len(mean_returns)
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1.0/num_assets] * num_assets)
    
    # Optimize for maximum Sharpe ratio
    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    max_sharpe_weights = max_sharpe_result['x']
    max_sharpe_return, max_sharpe_volatility = calculate_portfolio_performance(
        max_sharpe_weights, mean_returns, cov_matrix
    )
    max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility
    
    # Optimize for minimum volatility
    min_vol_result = minimize(
        min_portfolio_volatility,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    min_vol_weights = min_vol_result['x']
    min_vol_return, min_vol_volatility = calculate_portfolio_performance(
        min_vol_weights, mean_returns, cov_matrix
    )
    
    # Generate random portfolios for efficient frontier visualization
    results_array = np.zeros((3, num_portfolios))
    weights_array = np.zeros((num_portfolios, num_assets))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        weights_array[i, :] = weights
        
        # Calculate portfolio performance
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        
        # Store results
        results_array[0, i] = portfolio_return
        results_array[1, i] = portfolio_volatility
        results_array[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio
    
    return {
        'max_sharpe_weights': max_sharpe_weights,
        'max_sharpe_return': max_sharpe_return,
        'max_sharpe_volatility': max_sharpe_volatility,
        'max_sharpe_ratio': max_sharpe_ratio,
        'min_vol_weights': min_vol_weights,
        'min_vol_return': min_vol_return,
        'min_vol_volatility': min_vol_volatility,
        'results_array': results_array,
        'weights_array': weights_array
    }

def plot_efficient_frontier_plotly(results, risk_free_rate=0.01, asset_names=None):
    """Create an interactive Plotly visualization of the efficient frontier."""
    # Extract results
    max_sharpe_return = results['max_sharpe_return']
    max_sharpe_volatility = results['max_sharpe_volatility']
    min_vol_return = results['min_vol_return']
    min_vol_volatility = results['min_vol_volatility']
    results_array = results['results_array']
    
    # Create figure
    fig = go.Figure()
    
    # Add random portfolios scatter plot
    fig.add_trace(
        go.Scatter(
            x=results_array[1, :],
            y=results_array[0, :],
            mode='markers',
            marker=dict(
                size=5,
                color=results_array[2, :],
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                showscale=True
            ),
            text=[f'Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.2f}' 
                  for r, v, s in zip(results_array[0, :], results_array[1, :], results_array[2, :])],
            hoverinfo='text',
            name='Random Portfolios'
        )
    )
    
    # Add maximum Sharpe ratio portfolio
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_volatility],
            y=[max_sharpe_return],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='red',
                line=dict(width=2, color='black')
            ),
            text=f'Maximum Sharpe Ratio Portfolio<br>Return: {max_sharpe_return:.2%}<br>Volatility: {max_sharpe_volatility:.2%}<br>Sharpe: {results["max_sharpe_ratio"]:.2f}',
            hoverinfo='text',
            name='Maximum Sharpe Ratio'
        )
    )
    
    # Add minimum volatility portfolio
    fig.add_trace(
        go.Scatter(
            x=[min_vol_volatility],
            y=[min_vol_return],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='green',
                line=dict(width=2, color='black')
            ),
            text=f'Minimum Volatility Portfolio<br>Return: {min_vol_return:.2%}<br>Volatility: {min_vol_volatility:.2%}',
            hoverinfo='text',
            name='Minimum Volatility'
        )
    )
    
    # Add Capital Allocation Line (CAL)
    x_values = np.linspace(0, max(results_array[1, :]) * 1.2, 100)
    y_values = risk_free_rate + (max_sharpe_return - risk_free_rate) * x_values / max_sharpe_volatility
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Capital Allocation Line'
        )
    )
    
    # Add risk-free rate point
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[risk_free_rate],
            mode='markers',
            marker=dict(
                size=10,
                color='black'
            ),
            text=f'Risk-Free Rate: {risk_free_rate:.2%}',
            hoverinfo='text',
            name='Risk-Free Rate'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Efficient Frontier with Random Portfolios',
        xaxis=dict(
            title='Annualized Volatility (Standard Deviation)',
            tickformat='.0%',
            hoverformat='.2%'
        ),
        yaxis=dict(
            title='Annualized Expected Return',
            tickformat='.0%',
            hoverformat='.2%'
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        hovermode='closest',
        template='plotly_white',
        height=600
    )
    
    return fig

def plot_correlation_matrix_plotly(corr_matrix, asset_names):
    """Create an interactive Plotly visualization of the correlation matrix."""
    fig = px.imshow(
        corr_matrix,
        x=asset_names,
        y=asset_names,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        labels=dict(color='Correlation')
    )
    
    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{corr_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(
			color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                )
            )
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        height=600,
        template='plotly_white'
    )
    
    return fig

def plot_weights_plotly(weights, asset_names):
    """Create an interactive Plotly visualization of portfolio weights."""
    # Sort weights for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_assets = [asset_names[i] for i in sorted_indices]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=sorted_assets,
            y=sorted_weights * 100,
            marker=dict(
                color=sorted_weights,
                colorscale='Viridis',
                line=dict(width=1, color='black')
            ),
            text=[f"{w:.2%}" for w in sorted_weights],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Optimal Portfolio Weights',
        xaxis=dict(title='Assets'),
        yaxis=dict(
            title='Weight (%)',
            tickformat='.0f'
        ),
        template='plotly_white',
        height=500
    )
    
    return fig

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to calculate the portfolio's annualized risk (volatility)
def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
# Function to calculate the portfolio's annualized return
def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252
    
# Function to minimize the negative Sharpe ratio
def minimize_sharpe_ratio(weights, returns, risk_free_rate):
    return -(portfolio_return(weights, returns) - risk_free_rate) / portfolio_volatility(weights, returns)
    
# Function to calculate efficient frontier
def efficient_frontier_envelope(returns, risk_free_rate, num_portf=200):
    num_assets = returns.shape[1]
    init_guess = np.repeat(1 / num_assets, num_assets)
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Find the portfolio weights that maximize the Sharpe ratio
    optimal = minimize(minimize_sharpe_ratio, init_guess, args=(returns, risk_free_rate), bounds=bounds, constraints=constraints)
    optimal_weights = optimal.x
    # Calculate the target returns for the efficient frontier
    #target_returns = np.linspace(portfolio_return(init_guess, returns), portfolio_return(optimal_weights, returns), num_portfolios)
    min = returns.mean().min()*252
    max = returns.mean().max()*252
    target_returns = np.linspace(min, max, num_portf)
    target_volatilities = []
    weights_list = []
    
    for target_return in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_return})
        
        # Minimize portfolio volatility for each target return
        result = minimize(portfolio_volatility, init_guess, args=(returns,), bounds=bounds, constraints=constraints)
        target_volatilities.append(result.fun)
        weights_list.append(result.x)
    
    return np.array(target_volatilities), target_returns, weights_list, optimal_weights, optimal   
 
# Function to apply style based on a condition
def highlight_row(df, row_index, color='yellow'):
    """
    Changes the background color of an entire row in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        row_index (int): The index of the row to highlight.
        color (str, optional): The background color to apply. Defaults to 'yellow'.

    Returns:
         pd.io.formats.style.Styler: A Styler object with the applied style.
    """
    return df.style.apply(
        lambda x: ['background-color: {}'.format(color) if x.name == row_index else '' for i in x], axis=1
    )

# Display price history of selected stocks
if page == "Stocks Price History":
	st.markdown('<h2 class="sub-header">View stocks history of selected DOW stocks</h2>', unsafe_allow_html=True)
	view_stock_history("dowjones.csv")

# Main content based on selected page
if page == "Introduction":
    st.markdown('<h1 class="main-header">Modern Portfolio Theory Explorer\n\nAn Illustration Using Dow Jones Industrials</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="highlight">
        Modern Portfolio Theory (MPT) is a mathematical framework for assembling a portfolio of assets that maximizes expected return for a given level of risk. Developed by Harry Markowitz in 1952, MPT revolutionized investment management by providing a quantitative approach to portfolio construction and diversification.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">Key Concepts</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        1. **Risk and Return Trade-off**: Investors must accept higher risk to achieve higher expected returns.
        
        2. **Diversification Benefit**: Combining assets with imperfect correlations reduces portfolio risk without necessarily reducing expected returns.
        
        3. **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a defined level of risk.
        
        4. **Mean-Variance Optimization**: The mathematical process of finding portfolios on the efficient frontier.
        
        The central insight of MPT is that an asset's risk and return should not be evaluated in isolation, but by how it contributes to a portfolio's overall risk and return characteristics.
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/d/d4/Harry_Markowitz_%28Nobel%29.jpg", 
                 caption="Harry Markowitz, Nobel Laureate")
        
        st.markdown("""
        <div class="caption">
        Harry Markowitz was awarded the Nobel Memorial Prize in Economic Sciences in 1990 for his pioneering work on Modern Portfolio Theory.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Interactive Features</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This Streamlit application allows you to:
    
    - Explore the mathematical foundations of Modern Portfolio Theory
    - Generate and optimize portfolios with customizable parameters
    - Visualize the efficient frontier and correlation matrix
    - Analyze optimal portfolio weights
    - View the Python implementation of MPT
    
    Use the sidebar to navigate between different sections and adjust parameters for the interactive portfolio optimizer.
    """)
    
    st.markdown('<h2 class="sub-header">Historical Context</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Harry Markowitz published his groundbreaking paper "Portfolio Selection" in the Journal of Finance in 1952. Before Markowitz's work, investors focused primarily on assessing the risks and rewards of individual securities. MPT shifted the focus to how assets perform together in a portfolio.
    
    The theory was initially met with skepticism but gradually gained acceptance in both academic and professional investment communities. Today, MPT serves as the foundation for modern investment management and has influenced the development of index funds, asset allocation strategies, and risk management techniques.
    """)

elif page == "Mathematical Foundations":
    st.markdown('<h1 class="main-header">Mathematical Foundations of MPT</h1>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Expected Return</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    The expected return of a portfolio is the weighted average of the expected returns of the individual assets:
    
    <div class="formula">
    E(R<sub>p</sub>) = Σ w<sub>i</sub> E(R<sub>i</sub>)
    </div>
    
    Where:
    - E(R<sub>p</sub>) is the expected return of the portfolio
    - E(R<sub>i</sub>) is the expected return of asset i
    - w<sub>i</sub> is the weight (proportion) of asset i in the portfolio
    - Σ w<sub>i</sub> = 1 (weights sum to 1)
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Portfolio Risk</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    The risk of a portfolio is measured by its variance or standard deviation. The variance of a portfolio is:
    
    <div class="formula">
    σ<sub>p</sub><sup>2</sup> = Σ<sub>i</sub> Σ<sub>j</sub> w<sub>i</sub> w<sub>j</sub> σ<sub>i</sub> σ<sub>j</sub> ρ<sub>ij</sub>
    </div>
    
    Where:
    - σ<sub>p</sub><sup>2</sup> is the variance of the portfolio
    - σ<sub>i</sub> is the standard deviation of asset i
    - ρ<sub>ij</sub> is the correlation coefficient between assets i and j
    - w<sub>i</sub> and w<sub>j</sub> are the weights of assets i and j
    
    This can also be written using covariance:
    
    <div class="formula">
    σ<sub>p</sub><sup>2</sup> = Σ<sub>i</sub> Σ<sub>j</sub> w<sub>i</sub> w<sub>j</sub> σ<sub>ij</sub>
    </div>
    
    Where σ<sub>ij</sub> is the covariance between assets i and j.
    
    The standard deviation (volatility) of the portfolio is:
    
    <div class="formula">
    σ<sub>p</sub> = √σ<sub>p</sub><sup>2</sup>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Matrix Notation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    For computational efficiency, portfolio calculations are often expressed in matrix notation:
    
    - Let **w** be the vector of weights
    - Let **μ** be the vector of expected returns
    - Let **Σ** be the covariance matrix
    
    Then:
    - Portfolio expected return: E(R<sub>p</sub>) = **w**<sup>T</sup> **μ**
    - Portfolio variance: σ<sub>p</sub><sup>2</sup> = **w**<sup>T</sup> **Σ** **w**
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Diversification</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Diversification is a key concept in MPT. By holding combinations of assets that are not perfectly positively correlated, investors can reduce portfolio risk without necessarily sacrificing expected return.
    
    - If all asset pairs have correlations of 0 (perfectly uncorrelated), the portfolio's return variance is the sum of the weighted individual variances.
    - If all asset pairs have correlations of 1 (perfectly positively correlated), the portfolio's standard deviation is the weighted sum of the individual standard deviations.
    
    The benefit of diversification increases as the correlation between assets decreases.
    """)
    
    st.markdown('<h2 class="sub-header">Efficient Frontier</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    The efficient frontier represents the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return.
    
    In the risk-return space (with risk on the x-axis and return on the y-axis), the efficient frontier forms a hyperbolic curve. Each point on this curve represents a portfolio with the best possible expected return for its level of risk.
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/Markowitz_efficient_frontier.jpg", 
             caption="Efficient Frontier Illustration")
    
    st.markdown('<h2 class="sub-header">Capital Allocation Line (CAL)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    When a risk-free asset is introduced, the Capital Allocation Line (CAL) is formed. It's a straight line from the risk-free rate through a point on the efficient frontier, representing portfolios combining the risk-free asset and a risky portfolio.
    
    The slope of the CAL is the Sharpe ratio:
    
    <div class="formula">
    Sharpe Ratio = (E(R<sub>p</sub>) - R<sub>f</sub>) / σ<sub>p</sub>
    </div>
    
    Where R<sub>f</sub> is the risk-free rate.
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Portfolio Optimization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    The mathematical problem of finding the efficient frontier involves:
    
    1. Minimizing portfolio variance for a given expected return:
       
       Minimize: σ<sub>p</sub><sup>2</sup> = **w**<sup>T</sup> **Σ** **w**
       
       Subject to:
       - **w**<sup>T</sup> **μ** = target return
       - **w**<sup>T</sup> **1** = 1 (weights sum to 1)
    
    2. Maximizing the Sharpe ratio (when a risk-free asset is available):
       
       Maximize: (E(R<sub>p</sub>) - R<sub>f</sub>) / σ<sub>p</sub>
       
       Subject to:
       - **w**<sup>T</sup> **1** = 1 (weights sum to 1)
    
    These optimization problems can be solved using quadratic programming techniques.
    """, unsafe_allow_html=True)

elif page == "Interactive Portfolio Optimizer" and opt == True:
    st.markdown('<h1 class="main-header">Interactive Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
    This interactive tool allows you to generate and optimize portfolios based on Modern Portfolio Theory. 
    Adjust the parameters in the sidebar to customize the optimization process.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate data
    with st.spinner("Generating sample data and optimizing portfolio..."):
        returns_df, corr_matrix, prices = generate_sample_data(num_assets=num_assets)
        #st.session_state.my_checkbox = False

        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Optimize portfolio
        results = optimize_portfolio(mean_returns, cov_matrix, num_portfolios=num_portfolios, risk_free_rate=risk_free_rate)
    
    # Display tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Efficient Frontier", "Correlation Matrix", "Optimal Weights", "Asset Stats/Growth",
		"Stocks History"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Efficient Frontier</h3>', unsafe_allow_html=True)
        
        # Plot efficient frontier
        fig_ef = plot_efficient_frontier_plotly(results, risk_free_rate, returns_df.columns)
        st.plotly_chart(fig_ef, use_container_width=True)
        
        # Display optimization results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="section-header">Maximum Sharpe Ratio Portfolio</h4>', unsafe_allow_html=True)
            st.metric("Expected Return", f"{results['max_sharpe_return']:.2%}")
            st.metric("Volatility", f"{results['max_sharpe_volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{results['max_sharpe_ratio']:.4f}")
        
        with col2:
            st.markdown('<h4 class="section-header">Minimum Volatility Portfolio</h4>', unsafe_allow_html=True)
            st.metric("Expected Return", f"{results['min_vol_return']:.2%}")
            st.metric("Volatility", f"{results['min_vol_volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{(results['min_vol_return'] - risk_free_rate) / results['min_vol_volatility']:.4f}")
    
    with tab2:
        st.markdown('<h3 class="section-header">Correlation Matrix</h3>', unsafe_allow_html=True)
        
        # Plot correlation matrix
        fig_corr = plot_correlation_matrix_plotly(corr_matrix, returns_df.columns)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        <div class="caption">
        The correlation matrix shows the relationships between assets. Lower correlations (blue) indicate better diversification opportunities, 
        while higher correlations (red) indicate assets that move together.
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h3 class="section-header">Optimal Portfolio Weights</h3>', unsafe_allow_html=True)
        
        # Create tabs for different optimal portfolios
        weights_tab1, weights_tab2 = st.tabs(["Maximum Sharpe Ratio Weights", "Minimum Volatility Weights"])
        
        with weights_tab1:
            # Plot weights for maximum Sharpe ratio portfolio
            fig_weights_sharpe = plot_weights_plotly(results['max_sharpe_weights'], returns_df.columns)
            st.plotly_chart(fig_weights_sharpe, use_container_width=True)
            
            # Display weights as a table
            weights_df_sharpe = pd.DataFrame({
                'Asset': returns_df.columns,
                'Weight (%)': results['max_sharpe_weights'] * 100
            }).sort_values('Weight (%)', ascending=False)
            
            st.dataframe(weights_df_sharpe, use_container_width=True)
            
            # Download link
            st.markdown(get_download_link(weights_df_sharpe, 'max_sharpe_weights.csv', 'Download weights as CSV'), unsafe_allow_html=True)
        
        with weights_tab2:
            # Plot weights for minimum volatility portfolio
            fig_weights_min_vol = plot_weights_plotly(results['min_vol_weights'], returns_df.columns)
            st.plotly_chart(fig_weights_min_vol, use_container_width=True)
            
            # Display weights as a table
            weights_df_min_vol = pd.DataFrame({
                'Asset': returns_df.columns,
                'Weight (%)': results['min_vol_weights'] * 100
            }).sort_values('Weight (%)', ascending=False)
            
            st.dataframe(weights_df_min_vol, use_container_width=True)
            
            # Download link
            st.markdown(get_download_link(weights_df_min_vol, 'min_vol_weights.csv', 'Download weights as CSV'), unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h3 class="section-header">Asset Stats and Growth</h3>', unsafe_allow_html=True)
        
        # Display asset statistics
        stats_df = pd.DataFrame({
            'Asset': returns_df.columns,
            'Mean Daily Return (%)': mean_returns * 100,
            'Annual Return (%)': mean_returns * 252 * 100,
            'Daily Std Dev (%)': np.sqrt(np.diag(cov_matrix)) * 100,
            'Annual Std Dev (%)': np.sqrt(np.diag(cov_matrix)) * np.sqrt(252) * 100
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Download link
        st.markdown(get_download_link(stats_df, 'asset_statistics.csv', 'Download statistics as CSV'), unsafe_allow_html=True)
        
        # Display sample returns
        st.markdown('<h4 class="section-header">Sample Returns (First 10 Days)</h4>', unsafe_allow_html=True)
        st.dataframe(returns_df.head(10) * 100, use_container_width=True)
        st.markdown('<h4 class="section-header">Covariance Matrix</h4>', unsafe_allow_html=True)
        st.dataframe(returns_df.cov()*252 * 100, use_container_width=True)
        st.markdown('<h4 class="section-header">Correlation Matrix</h4>', unsafe_allow_html=True)
        st.dataframe(returns_df.corr(), use_container_width=True)
        
        target_volatilities, target_returns, weights_list, optimal_weights, optimal = \
			efficient_frontier_envelope(returns_df, risk_free_rate, num_portf=200)
			
        st.write("Efficient Frontier")
        plt.figure(figsize=(14,7))
        plt.plot(target_volatilities, target_returns)
        plt.grid(True)
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        
        # Calculate annualized returns and volatilities of individual stocks
        individual_returns = returns_df.mean() * 252
        individual_volatilities = returns_df.std() * np.sqrt(252)
        plt.scatter(individual_volatilities, individual_returns, marker='o', s=100)

        for i, txt in enumerate(prices.columns):
          plt.annotate(txt, (individual_volatilities[i], individual_returns[i]), fontsize=10, fontweight='bold', xytext=(10, 0), textcoords='offset points')

        # Save the efficient frontier and the associated portfolio weights in a pandas DataFrame
        ef_df = pd.DataFrame({'Volatility': target_volatilities, 'Expected Return': target_returns})
        weights_df = pd.DataFrame(data=weights_list, columns=prices.columns)
        sharpe_ratio = [(target_returns[i] - risk_free_rate)/target_volatilities[i] for i in range(len(target_returns))]
        sr = pd.DataFrame(sharpe_ratio, columns=['Sharpe Ratio'])        
        # st.write("Weights along Efficient Frontier")
        # st.dataframe(pd.concat([ef_df*100, sr, weights_df*100], axis=1).round(3), use_container_width=True)	

        max_index = sr.idxmax()
        r1 = target_returns[max_index]
        v1 = target_volatilities[max_index]
        plt.scatter(v1,r1, marker='X', s=100, color='red')
        txt1 = 'Max Sharpe Ratio'
        plt.annotate(txt1, (v1, r1), fontsize=10, fontweight='bold', xytext=(10, 0), textcoords='offset points') 
        
        min_index = ef_df['Volatility'].idxmin()
        r1 = target_returns[min_index]
        v1 = target_volatilities[min_index]
        plt.scatter(v1,r1, marker='X', s=100, color='red')
        txt1 = 'Minimum Volatility'
        plt.annotate(txt1, (v1, r1), fontsize=10, fontweight='bold', xytext=(10, 0), textcoords='offset points') 
        st.pyplot(plt)
            
        st.write("Weights(%) along Efficient Frontier")
        temp = pd.concat([ef_df*100, sr, weights_df*100], axis=1)
        #temp1 = highlight_row(temp, int(max_index), color='lightblue')
        st.write(temp.round(2), use_container_width=True)
        
        st.write("Portfolio Compositions(%) at Maximum Sharpe Ratio and Minimum Volatility")	
        selected_rows = temp.iloc[[int(max_index), int(min_index)]]
        selected_rows['Criteria'] = ['Max Sharpe \n Ratio', 'Minimum\nVolatitility']
        selected_rows.set_index('Criteria')
        st.write(selected_rows.round(2))
        
        #Monitor growth of portfolio

        stocks = symbols[0:num_assets]
        start_date = pd.to_datetime(d1)   #'2018-01-01'
        end_date = pd.to_datetime(d2)     #'2023-01-01'
        prices2 = pd.read_csv("dowjones.csv", index_col=0)
        prices2.index = pd.to_datetime(prices2.index)
        prices2 = prices2[(prices2.index > end_date)]
        
        prices2 = prices2[weights_df.columns]
        nshares = weights_df.iloc[int(max_index)]/prices2.iloc[0]
        initial_porfolio = np.dot(prices2.iloc[0], nshares)
        plt.figure(figsize=(10,6))
        plt.title("Porfolio Growth(%) - Since "  + str(prices2.index[0]))
        plt.grid(True)
        plt.xlabel("Time", fontsize=7)
        plt.ylabel("Portfolio Growth(%)", fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=6)	    
        plt.plot(prices2.index, 100.0*(np.dot(prices2, nshares) / initial_porfolio - 1.0), label='Maximum Sharpe Ratio')
        nshares = weights_df.iloc[int(min_index)]/prices2.iloc[0]
        initial_porfolio = np.dot(prices2.iloc[0], nshares)         
        plt.plot(prices2.index, 100.0*(np.dot(prices2, nshares) / initial_porfolio - 1.0), label='Minimum Volatility')
        plt.tick_params(axis='x', labelrotation=30)    
        plt.legend()
        st.pyplot(plt)
                
    with tab5:        
        plt.figure()
        prices.plot(linewidth=0.5, title="Stocks History", grid=True, fontsize=6)
        st.pyplot(plt)
        st.write("Stocks Price History : " + str(start_date) + " to " + str(end_date))
        st.write(prices.round(3))
        
        plt.figure()
        returns_df.plot(linewidth=0.5,  title="Daily Returns", grid=True)
        plt.savefig("returns.png")
        st.image("returns.png")
		
elif page == "Code Implementation":
    st.markdown('<h1 class="main-header">Python Implementation of MPT</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    Below is the Python code implementation of Modern Portfolio Theory. The code includes functions for generating sample data, 
    calculating portfolio performance, optimizing portfolios, and visualizing results.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different code sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Generation", "Portfolio Performance", "Optimization", "Visualization"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Sample Data Generation</h3>', unsafe_allow_html=True)
        
        st.code('''
def generate_sample_data(num_assets=10, num_observations=252):
    """Generate sample return data for assets."""
    # Create asset names
    asset_names = [f"Asset_{i+1}" for i in range(num_assets)]
    
    # Generate random returns with different means and standard deviations
    annual_returns = np.random.uniform(0.05, 0.15, num_assets)
    daily_returns = annual_returns / num_observations
    
    annual_volatilities = np.random.uniform(0.10, 0.30, num_assets)
    daily_volatilities = annual_volatilities / np.sqrt(num_observations)
    
    # Generate correlation matrix
    corr_matrix = np.random.uniform(0.1, 0.7, (num_assets, num_assets))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Ensure the matrix is positive-semidefinite
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        corr_matrix += np.eye(num_assets) * (abs(min_eig) + 1e-8)
    
    # Normalize to ensure it's a proper correlation matrix
    D = np.diag(1.0 / np.sqrt(np.diag(corr_matrix)))
    corr_matrix = D @ corr_matrix @ D
    
    # Generate returns using multivariate normal distribution
    cov_matrix = np.outer(daily_volatilities, daily_volatilities) * corr_matrix
    returns = np.random.multivariate_normal(
        mean=daily_returns,
        cov=cov_matrix,
        size=num_observations
    )
    
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    return returns_df, corr_matrix
        ''', language="python")
    
    with tab2:
        st.markdown('<h3 class="section-header">Portfolio Performance Calculation</h3>', unsafe_allow_html=True)
        
        st.code('''
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio performance metrics."""
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """Calculate the negative Sharpe ratio (for minimization)."""
    p_return, p_volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    
    return -sharpe_ratio

def min_portfolio_volatility(weights, mean_returns, cov_matrix):
    """Calculate portfolio volatility (for minimization)."""
    return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]
        ''', language="python")
    
    with tab3:
        st.markdown('<h3 class="section-header">Portfolio Optimization</h3>', unsafe_allow_html=True)
        
        st.code('''
def optimize_portfolio(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.01):
    """Optimize portfolio for maximum Sharpe ratio and minimum volatility."""
    num_assets = len(mean_returns)
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1.0/num_assets] * num_assets)
    
    # Optimize for maximum Sharpe ratio
    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    max_sharpe_weights = max_sharpe_result['x']
    max_sharpe_return, max_sharpe_volatility = calculate_portfolio_performance(
        max_sharpe_weights, mean_returns, cov_matrix
    )
    max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility
    
    # Optimize for minimum volatility
    min_vol_result = minimize(
        min_portfolio_volatility,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    min_vol_weights = min_vol_result['x']
    min_vol_return, min_vol_volatility = calculate_portfolio_performance(
        min_vol_weights, mean_returns, cov_matrix
    )
    
    # Generate random portfolios for efficient frontier visualization
    results_array = np.zeros((3, num_portfolios))
    weights_array = np.zeros((num_portfolios, num_assets))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        weights_array[i, :] = weights
        
        # Calculate portfolio performance
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        
        # Store results
        results_array[0, i] = portfolio_return
        results_array[1, i] = portfolio_volatility
        results_array[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio
    
    return {
        'max_sharpe_weights': max_sharpe_weights,
        'max_sharpe_return': max_sharpe_return,
        'max_sharpe_volatility': max_sharpe_volatility,
        'max_sharpe_ratio': max_sharpe_ratio,
        'min_vol_weights': min_vol_weights,
        'min_vol_return': min_vol_return,
        'min_vol_volatility': min_vol_volatility,
        'results_array': results_array,
        'weights_array': weights_array
    }
        ''', language="python")
    
    with tab4:
        st.markdown('<h3 class="section-header">Visualization with Plotly</h3>', unsafe_allow_html=True)
        
        st.code('''
def plot_efficient_frontier_plotly(results, risk_free_rate=0.01, asset_names=None):
    """Create an interactive Plotly visualization of the efficient frontier."""
    # Extract results
    max_sharpe_return = results['max_sharpe_return']
    max_sharpe_volatility = results['max_sharpe_volatility']
    min_vol_return = results['min_vol_return']
    min_vol_volatility = results['min_vol_volatility']
    results_array = results['results_array']
    
    # Create figure
    fig = go.Figure()
    
    # Add random portfolios scatter plot
    fig.add_trace(
        go.Scatter(
            x=results_array[1, :],
            y=results_array[0, :],
            mode='markers',
            marker=dict(
                size=5,
                color=results_array[2, :],
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                showscale=True
            ),
            text=[f'Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.2f}' 
                  for r, v, s in zip(results_array[0, :], results_array[1, :], results_array[2, :])],
            hoverinfo='text',
            name='Random Portfolios'
        )
    )
    
    # Add maximum Sharpe ratio portfolio
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_volatility],
            y=[max_sharpe_return],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='red',
                line=dict(width=2, color='black')
            ),
            text=f'Maximum Sharpe Ratio Portfolio<br>Return: {max_sharpe_return:.2%}<br>Volatility: {max_sharpe_volatility:.2%}<br>Sharpe: {results["max_sharpe_ratio"]:.2f}',
            hoverinfo='text',
            name='Maximum Sharpe Ratio'
        )
    )
        ''', language="python")
    
    st.markdown('<h3 class="section-header">Complete Code</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    The complete code for this Streamlit application is available for download. It includes all the functions shown above, 
    plus the Streamlit UI code to create this interactive web application.
    """)
    
    # Create a download button for the complete code
    with open(__file__, 'r') as f:
        code = f.read()
    
    b64 = base64.b64encode(code.encode()).decode()
    href = f'<a href="data:file/python;base64,{b64}" download="mpt_streamlit_app.py">Download Complete Code</a>'
    st.markdown(href, unsafe_allow_html=True)
    		   

elif page == "About":
    st.markdown('<h1 class="main-header">About This Application</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    This interactive Streamlit application was created to demonstrate Modern Portfolio Theory concepts and provide a tool for portfolio optimization.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Features</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    - **Educational Content**: Comprehensive explanation of Modern Portfolio Theory and its mathematical foundations
    - **Interactive Optimizer**: Generate and optimize portfolios with customizable parameters
    - **Dynamic Visualizations**: Interactive plots of the efficient frontier, correlation matrix, and portfolio weights
    - **Code Implementation**: Complete Python code for implementing MPT
    - **Data Export**: Download optimization results and asset statistics
    """)
    
    st.markdown('<h2 class="sub-header">Limitations</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    While Modern Portfolio Theory provides a powerful framework for portfolio construction, it has several limitations:
    
    1. **Assumes Normal Distribution**: MPT assumes returns follow a normal distribution, which often doesn't hold in real markets.
    2. **Based on Historical Data**: Using historical data to predict future performance has limitations.
    3. **Ignores Liquidity**: MPT doesn't account for asset liquidity constraints.
    4. **Simplifies Investor Behavior**: Assumes investors are rational and risk-averse.
    5. **Variance as Risk**: Uses variance as the sole measure of risk, ignoring downside risk.
    
    This application uses simulated data for demonstration purposes. In a real-world scenario, you would use actual historical returns data for your assets of interest.
    """)
    
    st.markdown('<h2 class="sub-header">References</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    1. Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.
    2. Investopedia. "Modern Portfolio Theory: What MPT Is and How Investors Use It." https://www.investopedia.com/terms/m/modernportfoliotheory.asp
    3. Wikipedia. "Modern portfolio theory." https://en.wikipedia.org/wiki/Modern_portfolio_theory
    """)
    
    st.markdown('<h2 class="sub-header">Technologies Used</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
        st.markdown("<div class='caption'>Streamlit</div>", unsafe_allow_html=True)
    
    with col2:
        st.image("https://numpy.org/images/logo.svg", width=100)
        st.markdown("<div class='caption'>NumPy</div>", unsafe_allow_html=True)
    
    with col3:
        st.image("https://pandas.pydata.org/static/img/pandas_mark.svg", width=100)
        st.markdown("<div class='caption'>Pandas</div>", unsafe_allow_html=True)
    
    with col4:
        st.image("https://matplotlib.org/stable/_images/sphx_glr_logos2_003.png", width=100)
        st.markdown("<div class='caption'>Matplotlib</div>", unsafe_allow_html=True)
    
    with col5:
        st.image("https://plotly.com/all_static/images/icon-dash.png", width=100)
        st.markdown("<div class='caption'>Plotly</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
    <p>Modern Portfolio Theory Explorer | Created with Streamlit</p>
    <p style="font-size: 0.8rem; color: #6c757d;">© 2025</p>
</div>
""", unsafe_allow_html=True)
