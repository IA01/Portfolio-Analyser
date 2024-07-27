import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from scipy import optimize
import requests
import plotly.graph_objects as go
import plotly.express as px
import io

# Set page config at the very beginning
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

def xnpv(rate, cashflows, dates):
    return sum(cf / (1 + rate)**((d - dates[0]).days / 365.0) for cf, d in zip(cashflows, dates))

def calculate_xirr(cashflows, dates):
    def f(rate):
        return xnpv(rate, cashflows, dates)
    return optimize.newton(f, 0.1)

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    history = stock.history(start=start_date, end=end_date)
    return history

def calculate_stock_metrics(transactions, symbol, end_date):
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
    df = df.sort_values('date')
    
    start_date = df['date'].min()
    history = get_stock_data(symbol, start_date, end_date)
    
    if history.empty:
        st.warning(f"No data available for {symbol}")
        return None
    
    current_price = history.iloc[-1]['Close']
    
    initial_investment = sum(row['quantity'] * row['price'] for _, row in df[df['quantity'] > 0].iterrows())
    sale_proceeds = sum(-row['quantity'] * row['price'] for _, row in df[df['quantity'] < 0].iterrows())
    
    current_quantity = df['quantity'].sum()
    current_value = current_quantity * current_price
    
    total_value = sale_proceeds + current_value
    total_return = total_value - initial_investment
    
    absolute_return = (total_return / initial_investment)
    
    df['cashflow'] = -df['quantity'] * df['price']
    df = pd.concat([df, pd.DataFrame({'date': [end_date], 'cashflow': [current_value]})], ignore_index=True)
    xirr = calculate_xirr(df['cashflow'], df['date'])
    
    holding_periods = []
    quantity = 0
    for _, row in df.iterrows():
        quantity += row['quantity'] if 'quantity' in row else 0
        if quantity > 0:
            holding_periods.append((row['date'], None))
        elif quantity == 0 and holding_periods and holding_periods[-1][1] is None:
            holding_periods[-1] = (holding_periods[-1][0], row['date'])
    
    if quantity > 0:
        holding_periods[-1] = (holding_periods[-1][0], end_date)
    
    returns = []
    for start, end in holding_periods:
        period_returns = history.loc[start:end]['Close'].pct_change().dropna()
        returns.extend(period_returns)
    
    annualized_std_dev = np.std(returns) * np.sqrt(252)  # Assuming 252 trading days in a year
    
    return {
        'symbol': symbol,
        'xirr': xirr,
        'absolute_return': absolute_return,
        'std_dev': annualized_std_dev,
        'current_value': current_value,
        'current_quantity': current_quantity,
        'current_price': current_price,
        'initial_investment': initial_investment,
        'sale_proceeds': sale_proceeds,
        'total_value': total_value,
        'total_return': total_return,
        'currency': '₹'
    }

def calculate_portfolio_metrics(portfolio):
    end_date = datetime.now(timezone.utc)
    results = []
    total_value = 0
    
    for symbol, stock_data in portfolio.items():
        transactions = stock_data['transactions']
        metrics = calculate_stock_metrics(transactions, symbol, end_date)
        if metrics:
            results.append(metrics)
            total_value += metrics['current_value']
    
    for result in results:
        result['weight'] = result['current_value'] / total_value
    
    portfolio_absolute_return = sum(r['absolute_return'] * r['weight'] for r in results)
    portfolio_xirr_return = sum(r['xirr'] * r['weight'] for r in results)
    portfolio_std_dev = np.sqrt(sum((r['std_dev'] * r['weight'])**2 for r in results))
    
    return results, portfolio_absolute_return, portfolio_xirr_return, portfolio_std_dev

def get_indian_stock_suggestions(query):
    url = f"https://www.nseindia.com/api/search/autocomplete?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        suggestions = [f"{item['symbol']} - {item['name']}" for item in data['symbols'] if item['symbol'].endswith('.NS')]
        return suggestions
    except:
        return []

def validate_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return 'symbol' in info and info['symbol'] == symbol
    except:
        return False

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #00A86B;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stock-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .stock-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stock-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .stock-title {
        font-size: 1.5em;
        font-weight: bold;
    }
    .stock-actions {
        display: flex;
        align-items: center;
    }
    .remove-stock {
        color: #ff4b4b;
        cursor: pointer;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Watermark
st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 10px; z-index: 1000; opacity: 0.7;'>
        Made by Ishaan Ahluwalia
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'editing' not in st.session_state:
    st.session_state.editing = False
if 'edit_stock' not in st.session_state:
    st.session_state.edit_stock = None
if 'edit_transaction' not in st.session_state:
    st.session_state.edit_transaction = None

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/BSE_LOGO.svg", width=100)  # BSE logo
    st.title("Portfolio Manager")

    st.subheader("Add Indian Stock")
    new_stock = st.text_input("Enter stock symbol", key="new_stock_input")
    suggestions = get_indian_stock_suggestions(new_stock)
    if suggestions:
        selected_stock = st.selectbox("Select from suggestions", suggestions, key="stock_suggestions")
        if selected_stock:
            new_stock = selected_stock.split(" - ")[0]
    
    if st.button("Add Stock", key="add_stock_button"):
        if new_stock:
            if not new_stock.endswith('.NS'):
                new_stock += '.NS'
            if validate_stock(new_stock):
                if new_stock not in st.session_state.portfolio:
                    st.session_state.portfolio[new_stock] = {'transactions': []}
                    st.success(f"Added {new_stock} to portfolio")
                else:
                    st.warning(f"{new_stock} already in portfolio")
            else:
                st.error(f"{new_stock} is not a valid NSE stock symbol")
        else:
            st.warning("Please enter a stock symbol")

    st.subheader("Risk-Free Rate")
    risk_free_rate = st.number_input("Enter risk-free rate (%)", min_value=0.0, max_value=100.0, value=7.0, step=0.1) / 100

# Main area
st.title("💸Portfolio Analyzer")

# Display current portfolio
st.header("📊 Current Portfolio")
if st.session_state.portfolio:
    for stock, data in st.session_state.portfolio.items():
        with st.expander(f"{stock} 🔍", expanded=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.subheader(stock)
            with col2:
                if st.button("Add Transaction", key=f"add_transaction_{stock}"):
                    st.session_state.editing = True
                    st.session_state.edit_stock = stock
                    st.session_state.edit_transaction = len(data['transactions'])
            with col3:
                if st.button("Delete Stock", key=f"remove_stock_{stock}"):
                    del st.session_state.portfolio[stock]
                    st.rerun()
            
            if data['transactions']:
                df = pd.DataFrame(data['transactions'])
                df['Total'] = df['quantity'] * df['price']
                st.dataframe(df.style.format({'price': '₹{:.2f}', 'Total': '₹{:.2f}'}))
                
                # Display current stock price and quantity held
                current_data = yf.Ticker(stock).history(period="1d")
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    current_quantity = df['quantity'].sum()
                    col1, col2 = st.columns(2)
                    col1.metric("Current Price", f"₹{current_price:.2f}")
                    col2.metric("Quantity Held", f"{current_quantity}")
                
                for i, transaction in enumerate(data['transactions']):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"Date: {transaction['date']}, Quantity: {transaction['quantity']}, Price: ₹{transaction['price']:.2f}")
                    with col2:
                        if st.button("Edit", key=f"edit_transaction_{stock}_{i}"):
                            st.session_state.editing = True
                            st.session_state.edit_stock = stock
                            st.session_state.edit_transaction = i
                    with col3:
                        if st.button("Delete", key=f"delete_transaction_{stock}_{i}"):
                            del data['transactions'][i]
                            st.rerun()
            else:
                st.info("No transactions yet")
else:
    st.info("Your portfolio is empty. Add some stocks to get started!")

# Transaction editing popup
if st.session_state.editing:
    with st.form("edit_transaction_form"):
        stock = st.session_state.edit_stock
        index = st.session_state.edit_transaction
        st.subheader(f"{'Edit' if index < len(st.session_state.portfolio[stock]['transactions']) else 'Add'} Transaction for {stock}")
        
        st.info("Use positive quantity for buying and negative quantity for selling.")
        
        if index < len(st.session_state.portfolio[stock]['transactions']):
            transaction = st.session_state.portfolio[stock]['transactions'][index]
            date = st.date_input("Date", value=datetime.strptime(transaction['date'], '%Y-%m-%d').date())
            quantity = st.number_input("Quantity (positive for buy, negative for sell)", value=transaction['quantity'], step=1)
            price = st.number_input("Price", value=transaction['price'], min_value=0.01, step=0.01)
        else:
            date = st.date_input("Date")
            quantity = st.number_input("Quantity (positive for buy, negative for sell)", step=1)
            price = st.number_input("Price", min_value=0.01, step=0.01)
        
        submitted = st.form_submit_button("Save")
        if submitted:
            new_transaction = {
                'date': date.strftime('%Y-%m-%d'),
                'quantity': quantity,
                'price': price
            }
            if index < len(st.session_state.portfolio[stock]['transactions']):
                st.session_state.portfolio[stock]['transactions'][index] = new_transaction
            else:
                st.session_state.portfolio[stock]['transactions'].append(new_transaction)
            st.session_state.editing = False
            st.rerun()
        
        if st.form_submit_button("Cancel"):
            st.session_state.editing = False
            st.rerun()

# Calculate metrics button
if st.button("📈 Calculate Portfolio Metrics", key="calculate_metrics_button"):
    if st.session_state.portfolio:
        results, portfolio_absolute_return, portfolio_xirr_return, portfolio_std_dev = calculate_portfolio_metrics(st.session_state.portfolio)
        
        # Portfolio Overview
        st.header("📊 Portfolio Overview")
        
        # Pie Chart
        fig_pie = px.pie(
            values=[r['current_value'] for r in results],
            names=[r['symbol'] for r in results],
            title="Portfolio Composition",
            hole=0.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie)

          # Scatter Plot
        fig_scatter = px.scatter(
            x=[r['xirr'] * 100 for r in results],
            y=[r['std_dev'] * 100 for r in results],
            text=[r['symbol'] for r in results],
            labels={'x': 'XIRR (%)', 'y': 'Standard Deviation (%)'},
            title="XIRR vs Standard Deviation"
        )
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter)

        # Individual Stock Metrics
        st.header("🏢 Individual Stock Metrics")
        metrics_df = pd.DataFrame(results)
        metrics_df = metrics_df[['symbol', 'xirr', 'absolute_return', 'std_dev', 'current_value', 'current_quantity', 'current_price', 'weight']]
        metrics_df.columns = ['Symbol', 'XIRR', 'Absolute Return', 'Std Dev', 'Current Value', 'Current Quantity', 'Current Price', 'Weight']
        metrics_df['XIRR'] = metrics_df['XIRR'].apply(lambda x: f"{x*100:.2f}%")
        metrics_df['Absolute Return'] = metrics_df['Absolute Return'].apply(lambda x: f"{x*100:.2f}%")
        metrics_df['Std Dev'] = metrics_df['Std Dev'].apply(lambda x: f"{x*100:.2f}%")
        metrics_df['Current Value'] = metrics_df['Current Value'].apply(lambda x: f"₹{x:.2f}")
        metrics_df['Current Price'] = metrics_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
        metrics_df['Weight'] = metrics_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(metrics_df)

        # Portfolio Metrics
        st.header("💼 Portfolio Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Absolute Return", f"{portfolio_absolute_return*100:.2f}%")
            st.metric("Portfolio XIRR-based Return", f"{portfolio_xirr_return*100:.2f}%")
            st.metric("Portfolio Standard Deviation", f"{portfolio_std_dev*100:.2f}%")
        with col2:
            sharpe_ratio_absolute = (portfolio_absolute_return - risk_free_rate) / portfolio_std_dev
            sharpe_ratio_xirr = (portfolio_xirr_return - risk_free_rate) / portfolio_std_dev
            st.metric("Sharpe Ratio (Absolute Return)", f"{sharpe_ratio_absolute:.2f}")
            st.metric("Sharpe Ratio (XIRR-based)", f"{sharpe_ratio_xirr:.2f}")

    else:
        st.warning("Please add stocks and transactions to your portfolio first")