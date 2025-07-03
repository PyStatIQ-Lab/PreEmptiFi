import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

# Sample client data (would normally come from database/API)
CLIENT_DATA = pd.DataFrame([
    {"Client Name": "Client_1", "Equity Portfolio": "Reliance, Sensex ETF", "Mutual Fund Holdings": "ICICI Pru Equity Fund, Mirae Emerging Bluechip", "Total Portfolio Size": 95.95, "Risk Profile Score": 9, "Risk Category": "Very Aggressive"},
    {"Client Name": "Client_5", "Equity Portfolio": "HDFC Bank, NIFTY ETF", "Mutual Fund Holdings": "ICICI Pru Equity Fund, Axis Midcap Fund", "Total Portfolio Size": 88.78, "Risk Profile Score": 1, "Risk Category": "Very Conservative"},
    {"Client Name": "Client_22", "Equity Portfolio": "TCS, NIFTY ETF", "Mutual Fund Holdings": "ICICI Pru Equity Fund, Axis Midcap Fund", "Total Portfolio Size": 72.67, "Risk Profile Score": 5, "Risk Category": "Moderate"}
])

# Stock beta values (would normally come from market data API)
STOCK_BETAS = {
    'Reliance': 1.15, 'HDFC Bank': 1.2, 'TCS': 0.9, 
    'Infosys': 1.05, 'Sensex ETF': 1.0, 'NIFTY ETF': 1.0
}

# Sector mapping for stocks
STOCK_SECTORS = {
    'Reliance': 'Energy', 'HDFC Bank': 'Financial', 'TCS': 'IT', 
    'Infosys': 'IT', 'Sensex ETF': 'Diversified', 'NIFTY ETF': 'Diversified'
}

# Cache news data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Refresh every hour
def fetch_news_data():
    try:
        url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=10"
        response = requests.get(url, timeout=10)
        return response.json()['data']
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Simple sentiment analysis (in production would use NLP model)
def analyze_sentiment(headline):
    negative_keywords = ['hike', 'war', 'fall', 'drop', 'crisis', 'cut', 'loss']
    positive_keywords = ['growth', 'gain', 'profit', 'rise', 'boom']
    
    if any(word in headline.lower() for word in negative_keywords):
        return "Negative"
    elif any(word in headline.lower() for word in positive_keywords):
        return "Positive"
    return "Neutral"

# Extract affected sectors from news (simplified)
def extract_affected_sectors(headline):
    sector_map = {
        'RBI': ['Financial'],
        'rate': ['Financial', 'Real Estate'],
        'IT': ['IT'],
        'tech': ['IT'],
        'oil': ['Energy'],
        'monsoon': ['FMCG', 'Agriculture']
    }
    
    affected = []
    for keyword, sectors in sector_map.items():
        if keyword.lower() in headline.lower():
            affected.extend(sectors)
    return list(set(affected)) if affected else ['General Market']

# Identify high-risk clients based on news
def flag_high_risk_clients(news_data, client_data):
    high_risk_clients = []
    
    for news_item in news_data:
        headline = news_item.get('title', '')
        sentiment = analyze_sentiment(headline)
        affected_sectors = extract_affected_sectors(headline)
        
        if sentiment != "Negative":
            continue
            
        for _, client in client_data.iterrows():
            # Get all stocks in client's portfolio
            client_stocks = []
            if isinstance(client['Equity Portfolio'], str):
                client_stocks.extend([s.strip() for s in client['Equity Portfolio'].split(',')])
            if isinstance(client['Mutual Fund Holdings'], str):
                client_stocks.extend([s.strip() for s in client['Mutual Fund Holdings'].split(',')])
            
            # Check if any stock matches affected sectors
            exposed_stocks = []
            for stock in client_stocks:
                if STOCK_SECTORS.get(stock, '') in affected_sectors:
                    exposed_stocks.append(stock)
            
            if exposed_stocks:
                high_risk_clients.append({
                    'Client Name': client['Client Name'],
                    'Risk Category': client['Risk Category'],
                    'Exposed Stocks': ', '.join(exposed_stocks),
                    'News Impact': headline,
                    'Sentiment': sentiment,
                    'Affected Sectors': ', '.join(affected_sectors)
                })
    
    return pd.DataFrame(high_risk_clients)

# Simulate portfolio impact
def simulate_shock(client, shock_percent, stock_betas):
    if not isinstance(client['Equity Portfolio'], str):
        return {}
    
    portfolio_value = client['Total Portfolio Size']
    equity_allocation = 0.7  # Assuming 70% in equities (simplified)
    equity_value = portfolio_value * equity_allocation
    
    stocks = [s.strip() for s in client['Equity Portfolio'].split(',')]
    shock_results = {}
    total_impact = 0
    
    for stock in stocks:
        beta = stock_betas.get(stock, 1.0)
        stock_impact = beta * shock_percent
        shock_results[stock] = {
            'Beta': beta,
            'Estimated Impact (%)': stock_impact,
            'Value Impact (₹L)': round(equity_value * (len(stocks)**-1 * stock_impact/100), 2)
        }
        total_impact += shock_results[stock]['Value Impact (₹L)']
    
    shock_results['Total Portfolio Impact (₹L)'] = round(total_impact, 2)
    shock_results['Total Portfolio Impact (%)'] = round((total_impact/portfolio_value)*100, 2)
    
    return shock_results

# Generate mitigation recommendations
def generate_recommendations(client, shock_results):
    recommendations = []
    risk_category = client['Risk Category']
    
    if risk_category in ['Very Aggressive', 'Aggressive']:
        recommendations.append("Consider hedging with index options (Nifty/Bank Nifty puts)")
        recommendations.append("Review stop-loss levels for high-beta stocks")
    elif risk_category in ['Moderate', 'Conservative']:
        recommendations.append("Increase allocation to debt instruments (10-15%)")
        recommendations.append("Switch from sector funds to diversified funds")
    else:  # Very Conservative
        recommendations.append("Move 20% to liquid funds temporarily")
        recommendations.append("Activate capital protection strategies")
    
    if shock_results.get('Total Portfolio Impact (%)', 0) > 5:
        recommendations.append(f"Immediate rebalancing suggested (>5% impact)")
    
    return recommendations

# Main Streamlit app
def main():
    st.title("AI-Powered Portfolio Risk Management")
    st.subheader("Real-time News Impact Analysis & What-If Scenarios")
    
    # Fetch market news
    with st.spinner("Fetching latest market news..."):
        news_data = fetch_news_data()
    
    if not news_data:
        st.warning("Could not fetch news data. Using sample data.")
        news_data = [
            {"title": "RBI hikes repo rate by 50 bps unexpectedly", "createdAt": "2024-07-04T10:30:00Z"},
            {"title": "US-China trade war escalates, IT stocks fall", "createdAt": "2024-07-04T09:15:00Z"}
        ]
    
    # Display critical news
    st.subheader("🚨 Market-Moving News")
    for news in news_data[:3]:  # Show top 3
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{news.get('title', '')}**")
        with col2:
            sentiment = analyze_sentiment(news.get('title', ''))
            st.write(f"_{sentiment}_")
    
    # Risk analysis section
    st.subheader("🔍 High-Risk Client Identification")
    high_risk_df = flag_high_risk_clients(news_data, CLIENT_DATA)
    
    if not high_risk_df.empty:
        st.dataframe(high_risk_df[['Client Name', 'Risk Category', 'Exposed Stocks', 'Affected Sectors']])
        
        # Select client for detailed analysis
        selected_client_name = st.selectbox("Select client for detailed analysis:", high_risk_df['Client Name'].unique())
        selected_client = CLIENT_DATA[CLIENT_DATA['Client Name'] == selected_client_name].iloc[0]
        
        st.subheader(f"📊 Portfolio Impact Simulation for {selected_client_name}")
        
        # Shock scenario selection
        shock_scenario = st.radio("Select shock scenario:", 
                                 ["Moderate (-5%)", "Severe (-10%)", "Custom"])
        
        if shock_scenario == "Moderate (-5%)":
            shock_percent = -5.0
        elif shock_scenario == "Severe (-10%)":
            shock_percent = -10.0
        else:
            shock_percent = st.number_input("Enter custom shock percentage:", -30.0, 0.0, -7.5)
        
        # Run simulation
        shock_results = simulate_shock(selected_client, shock_percent, STOCK_BETAS)
        
        if shock_results:
            st.write("### Portfolio Impact Summary")
            impact_df = pd.DataFrame.from_dict({
                k: v for k, v in shock_results.items() 
                if not isinstance(v, dict)
            }, orient='index', columns=['Value'])
            st.dataframe(impact_df)
            
            # Show stock-level impacts
            st.write("### Stock-Level Impacts")
            stock_impacts = pd.DataFrame.from_dict({
                k: v for k, v in shock_results.items() 
                if isinstance(v, dict)
            }, orient='index')
            st.dataframe(stock_impacts)
            
            # Generate recommendations
            st.write("### 🛡️ Risk Mitigation Recommendations")
            recommendations = generate_recommendations(selected_client, shock_results)
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Visualization
            st.write("### Impact Visualization")
            fig, ax = plt.subplots()
            stock_impacts['Estimated Impact (%)'].plot(kind='bar', ax=ax, color='red')
            ax.set_title("Estimated Impact by Holding")
            ax.set_ylabel("Percentage Impact")
            st.pyplot(fig)
    else:
        st.success("No high-risk clients identified based on current news.")

if __name__ == "__main__":
    main()
