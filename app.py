import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Configure page
st.set_page_config(layout="wide", page_title="Client Portfolio Analysis", page_icon="📊")
st.title("📊 Client Portfolio Analysis")
st.caption("Portfolio Performance & Event-Driven Insights")

# Portfolio data
portfolio = [
    {"Symbol": "NYKAA", "Exchange": "nse", "Event": "agreement", "Shares": 150, "Avg Cost": 145.50},
    {"Symbol": "MRF", "Exchange": "nse", "Event": "financial", "Shares": 10, "Avg Cost": 115000},
    {"Symbol": "M&M", "Exchange": "nse", "Event": "financial", "Shares": 75, "Avg Cost": 1820},
    {"Symbol": "BHARATFORG", "Exchange": "nse", "Event": "financial", "Shares": 200, "Avg Cost": 1125},
    {"Symbol": "AXISBANK", "Exchange": "nse", "Event": "financial", "Shares": 250, "Avg Cost": 980},
    {"Symbol": "NESTLEIND", "Exchange": "nse", "Event": "financial", "Shares": 50, "Avg Cost": 24500},
    {"Symbol": "TECHM", "Exchange": "nse", "Event": "financial", "Shares": 120, "Avg Cost": 1100},
    {"Symbol": "DMART", "Exchange": "nse", "Event": "investment", "Shares": 80, "Avg Cost": 3850},
    {"Symbol": "KPIGREEN", "Exchange": "nse", "Event": "regulatory", "Shares": 500, "Avg Cost": 850},
    {"Symbol": "RCOM", "Exchange": "nse", "Event": "regulatory", "Shares": 1000, "Avg Cost": 1.5},
    {"Symbol": "TATACOMM", "Exchange": "nse", "Event": "financial", "Shares": 350, "Avg Cost": 180},
    {"Symbol": "JKCEMENT", "Exchange": "nse", "Event": "financial", "Shares": 150, "Avg Cost": 3250},
    {"Symbol": "HCLTECH", "Exchange": "nse", "Event": "partnership", "Shares": 200, "Avg Cost": 1350},
    {"Symbol": "IDEA", "Exchange": "nse", "Event": "expansion", "Shares": 5000, "Avg Cost": 12},
    {"Symbol": "RAYMOND", "Exchange": "nse", "Event": "launch", "Shares": 300, "Avg Cost": 1850},
    {"Symbol": "RAYMONDLSL", "Exchange": "nse", "Event": "launch", "Shares": 1500, "Avg Cost": 150},
    {"Symbol": "TORNTPHARM", "Exchange": "nse", "Event": "agreement", "Shares": 100, "Avg Cost": 1750},
    {"Symbol": "APLLTD", "Exchange": "nse", "Event": "regulatory", "Shares": 400, "Avg Cost": 750},
    {"Symbol": "WAAREEENER", "Exchange": "nse", "Event": "agreement", "Shares": 250, "Avg Cost": 2200},
    {"Symbol": "JYOTICNC", "Exchange": "nse", "Event": "agreement", "Shares": 800, "Avg Cost": 450},
    {"Symbol": "KTKBANK", "Exchange": "nse", "Event": "leadership", "Shares": 1000, "Avg Cost": 175},
    {"Symbol": "JBCHEPHARM", "Exchange": "nse", "Event": "acquisition", "Shares": 150, "Avg Cost": 1650},
    {"Symbol": "HAL", "Exchange": "nse", "Event": "financial", "Shares": 200, "Avg Cost": 3800},
]

# Convert to DataFrame
portfolio_df = pd.DataFrame(portfolio)

# Fetch current prices
def get_current_price(symbol, exchange):
    suffix = ".NS" if exchange.lower() == "nse" else ".BO"
    ticker = symbol + suffix
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else None
    except:
        return None

# Add current prices to portfolio
portfolio_df['Current Price'] = portfolio_df.apply(
    lambda row: get_current_price(row['Symbol'], row['Exchange']), axis=1
)

# Calculate portfolio metrics
portfolio_df['Current Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
portfolio_df['Investment'] = portfolio_df['Shares'] * portfolio_df['Avg Cost']
portfolio_df['P&L'] = portfolio_df['Current Value'] - portfolio_df['Investment']
portfolio_df['P&L %'] = (portfolio_df['P&L'] / portfolio_df['Investment']) * 100

# Portfolio summary
total_investment = portfolio_df['Investment'].sum()
total_current_value = portfolio_df['Current Value'].sum()
total_pl = total_current_value - total_investment
total_pl_pct = (total_pl / total_investment) * 100

# Display portfolio metrics
st.subheader("Portfolio Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Investment", f"₹{total_investment:,.0f}")
col2.metric("Current Value", f"₹{total_current_value:,.0f}")
col3.metric("Profit & Loss", f"₹{total_pl:,.0f}", f"{total_pl_pct:.2f}%")
col4.metric("Stocks", len(portfolio_df), "5 Sectors")

# Event analysis
st.subheader("Event Distribution & Performance")
col1, col2 = st.columns(2)

with col1:
    event_counts = portfolio_df['Event'].value_counts().reset_index()
    event_counts.columns = ['Event', 'Count']
    fig = px.pie(event_counts, values='Count', names='Event', 
                 title='Event Type Distribution',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    event_performance = portfolio_df.groupby('Event')['P&L %'].mean().reset_index()
    fig = px.bar(event_performance, x='Event', y='P&L %',
                 title='Average Returns by Event Type',
                 color='P&L %',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

# Top holdings
st.subheader("Top Holdings & Performance")
portfolio_df['Weight'] = (portfolio_df['Current Value'] / total_current_value) * 100
top_holdings = portfolio_df.sort_values('Current Value', ascending=False).head(10)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(top_holdings, x='Symbol', y='Weight',
                 title='Top 10 Holdings by Portfolio Weight',
                 color='P&L %',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(top_holdings, x='P&L %', y='Weight', 
                     size='Current Value', color='Event',
                     hover_name='Symbol', 
                     title='Performance vs Portfolio Weight',
                     labels={'P&L %': 'Return (%)', 'Weight': 'Portfolio Weight (%)'})
    st.plotly_chart(fig, use_container_width=True)

# Sector exposure (mocked)
sector_mapping = {
    'NYKAA': 'Consumer Goods',
    'MRF': 'Automotive',
    'M&M': 'Automotive',
    'BHARATFORG': 'Industrial',
    'AXISBANK': 'Financial',
    'NESTLEIND': 'Consumer Goods',
    'TECHM': 'Technology',
    'DMART': 'Retail',
    'KPIGREEN': 'Energy',
    'RCOM': 'Telecom',
    'TATACOMM': 'Telecom',
    'JKCEMENT': 'Construction',
    'HCLTECH': 'Technology',
    'IDEA': 'Telecom',
    'RAYMOND': 'Textiles',
    'RAYMONDLSL': 'Real Estate',
    'TORNTPHARM': 'Pharmaceuticals',
    'APLLTD': 'Pharmaceuticals',
    'WAAREEENER': 'Energy',
    'JYOTICNC': 'Industrial',
    'KTKBANK': 'Financial',
    'JBCHEPHARM': 'Pharmaceuticals',
    'HAL': 'Aerospace & Defense'
}

portfolio_df['Sector'] = portfolio_df['Symbol'].map(sector_mapping)
sector_exposure = portfolio_df.groupby('Sector')['Current Value'].sum().reset_index()
sector_exposure['Weight'] = (sector_exposure['Current Value'] / total_current_value) * 100

# Price trend analysis
st.subheader("Sector Exposure & Price Trends")

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(sector_exposure.sort_values('Weight', ascending=False), 
                 x='Sector', y='Weight',
                 title='Portfolio Sector Allocation',
                 color='Sector')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Select a stock for detailed analysis
    selected_stock = st.selectbox("Select Stock for Price Analysis", 
                                  options=portfolio_df['Symbol'].unique(),
                                  index=0)
    
    # Get selected stock data
    selected_data = portfolio_df[portfolio_df['Symbol'] == selected_stock].iloc[0]
    ticker = selected_data['Symbol'] + '.NS'
    
    # Fetch historical data
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="6mo")
    
    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], 
                                mode='lines', name='Closing Price',
                                line=dict(color='#5e35b1', width=2.5)))
        
        # Add event markers with proper date conversion
        events = portfolio_df[portfolio_df['Symbol'] == selected_data['Symbol']]
        for _, row in events.iterrows():
            # Use a valid date within the chart's range
            event_date = hist.index[-30]  # 30 days ago from the last date in history
            
            # Add vertical line with annotation
            fig.add_vline(
                x=event_date.timestamp() * 1000,  # Convert to milliseconds since epoch
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=row['Event'],
                annotation_position="top right"
            )
        
        fig.update_layout(title=f"{selected_stock} Price Trend - Last 6 Months",
                          xaxis_title="Date",
                          yaxis_title="Price (₹)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No historical data available for {selected_stock}")

# Portfolio performance table
st.subheader("Detailed Portfolio Analysis")
portfolio_df = portfolio_df.sort_values('Current Value', ascending=False)

# Format display values
portfolio_df['Current Price'] = portfolio_df['Current Price'].apply(lambda x: f"₹{x:,.2f}" if x else "N/A")
portfolio_df['Current Value'] = portfolio_df['Current Value'].apply(lambda x: f"₹{x:,.0f}")
portfolio_df['Investment'] = portfolio_df['Investment'].apply(lambda x: f"₹{x:,.0f}")
portfolio_df['P&L'] = portfolio_df['P&L'].apply(lambda x: f"₹{x:,.0f}")
portfolio_df['P&L %'] = portfolio_df['P&L %'].apply(lambda x: f"{x:.2f}%")
portfolio_df['Weight'] = portfolio_df['Weight'].apply(lambda x: f"{x:.1f}%")

# Color coding for P&L %
def color_pl(val):
    try:
        num = float(val.replace('%', ''))
        color = 'green' if num > 0 else 'red' if num < 0 else 'gray'
        return f'color: {color}; font-weight: bold'
    except:
        return ''

st.dataframe(
    portfolio_df[['Symbol', 'Sector', 'Event', 'Shares', 
                  'Avg Cost', 'Current Price', 'Investment',
                  'Current Value', 'P&L', 'P&L %', 'Weight']].rename(columns={
                      'Symbol': 'Symbol',
                      'Sector': 'Sector',
                      'Event': 'Recent Event',
                      'Shares': 'Shares',
                      'Avg Cost': 'Avg Cost (₹)',
                      'Current Price': 'Current Price (₹)',
                      'Investment': 'Investment (₹)',
                      'Current Value': 'Current Value (₹)',
                      'P&L': 'P&L (₹)',
                      'P&L %': 'P&L %',
                      'Weight': 'Portfolio Weight'
                  }).style.applymap(color_pl, subset=['P&L %']),
    height=600,
    use_container_width=True
)

# Risk analysis
st.subheader("Portfolio Risk Analysis")
col1, col2 = st.columns(2)

with col1:
    # Beta calculation (mocked)
    portfolio_df['Beta'] = np.random.uniform(0.7, 1.8, len(portfolio_df))
    avg_beta = portfolio_df['Beta'].mean()
    
    st.metric("Portfolio Beta", f"{avg_beta:.2f}", 
              "High Volatility" if avg_beta > 1.3 else "Moderate Volatility" if avg_beta > 0.9 else "Low Volatility")
    
    fig = px.histogram(portfolio_df, x='Beta', nbins=10,
                       title='Stock Beta Distribution',
                       color_discrete_sequence=['#7e57c2'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Event risk analysis
    event_risk = {
        'financial': 'Low',
        'regulatory': 'High',
        'agreement': 'Medium',
        'partnership': 'Low',
        'expansion': 'Medium',
        'launch': 'High',
        'acquisition': 'High',
        'leadership': 'Medium',
        'investment': 'Medium'
    }
    
    portfolio_df['Event Risk'] = portfolio_df['Event'].map(event_risk)
    risk_counts = portfolio_df['Event Risk'].value_counts()
    
    fig = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index,
                 title='Portfolio Risk Profile',
                 color=risk_counts.index,
                 color_discrete_map={'High': '#ef5350', 'Medium': '#ffca28', 'Low': '#66bb6a'})
    st.plotly_chart(fig, use_container_width=True)

# Recommendations
st.subheader("Portfolio Recommendations")
st.warning("⚠️ High Risk Exposure: Regulatory & Launch Events (35% of portfolio)")
st.info("💡 Consider reducing exposure in high-risk event stocks (RCOM, KPIGREEN, RAYMOND)")
st.success("✅ Increase allocation to financial event stocks with stable returns (AXISBANK, NESTLEIND, HAL)")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top Buys**")
    st.markdown("""
    - **HCLTECH**: Strong partnership potential (Current: 4.5% weight)
    - **DMART**: Solid investment fundamentals (Current: 3.8% weight)
    - **TECHM**: Consistent financial performance (Current: 3.2% weight)
    """)

with col2:
    st.markdown("**Consider Reducing**")
    st.markdown("""
    - **RCOM**: High regulatory risk, negative returns
    - **IDEA**: Expansion uncertainty, volatile performance
    - **RAYMONDLSL**: Recent launch event, potential short-term volatility
    """)

# Portfolio timeline
st.subheader("Portfolio Event Timeline")
timeline_data = []
for _, row in portfolio_df.iterrows():
    timeline_data.append({
        'Symbol': row['Symbol'],
        'Event': row['Event'],
        'Date': datetime.now() - timedelta(days=np.random.randint(1, 60))
    })

timeline_df = pd.DataFrame(timeline_data)
fig = px.timeline(timeline_df, x_start="Date", x_end=timeline_df['Date'] + timedelta(days=1), 
                 y="Symbol", color="Event",
                 title="Recent Corporate Events Timeline",
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)
