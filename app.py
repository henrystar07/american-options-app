import streamlit as st
import numpy as np
import os
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from numba import jit
import requests
from functools import lru_cache
from retrying import retry

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:49633"  # HTTP代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:49633"  # HTTP代理
proxies = {
    "http": os.environ["HTTP_PROXY"],
    "https": os.environ["HTTPS_PROXY"]
}
@jit(nopython=True, cache=True)
def american_option_price(S, K, T, r, q, sigma, option_type, steps=100):
    dt = T/steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt) - d)/(u - d)
    
    price_tree = np.zeros((steps+1, steps+1))
    price_tree[0,0] = S
    for i in range(1, steps+1):
        price_tree[i,0] = price_tree[i-1,0] * u
        for j in range(1, i+1):
            price_tree[i,j] = price_tree[i-1,j-1] * d
    
    option_tree = np.zeros_like(price_tree)
    if option_type == 'call':
        option_tree[-1] = np.maximum(price_tree[-1] - K, 0)
    else:
        option_tree[-1] = np.maximum(K - price_tree[-1], 0)
    
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            exercise = max(price_tree[i,j] - K, 0) if option_type=='call' else max(K - price_tree[i,j],0)
            hold = np.exp(-r*dt) * (p*option_tree[i+1,j] + (1-p)*option_tree[i+1,j+1])
            option_tree[i,j] = max(exercise, hold)
    
    return option_tree[0,0]


@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
def fetch_price(ticker):
    """增强型数据获取函数"""
    try:
        # 方法1：使用yfinance官方API
        stock = yf.Ticker(ticker, proxy=os.environ["HTTPS_PROXY"])
        hist = stock.history(period="1d", timeout=10)
        if not hist.empty:
            return hist['Close'].iloc[-1], stock.info.get('currency', 'USD')
        
        # 方法2：使用备用API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": "1d", "interval": "1d"}
        response = requests.get(url, params=params, timeout=10, proxies=proxies)
        data = response.json()
        return data['chart']['result'][0]['meta']['regularMarketPrice'], 'USD'
        
    except Exception as e:
        # 方法3：使用Alpha Vantage备用源
        av_url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": st.secrets["ALPHAVANTAGE_KEY"]
        }
        response = requests.get(av_url, params=params, proxies=proxies)
        data = response.json()
        return float(data['Global Quote']['05. price']), 'USD'

def main():
    st.set_page_config(page_title="美式期权分析系统", layout="wide", page_icon="📈")
    
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("参数设置")
        ticker = st.text_input("标的代码", value="AAPL")
        option_type = st.selectbox("期权类型", ["Call", "Put"])
        col1, col2 = st.columns(2)
        with col1:
            K = st.number_input("行权价", value=100.0)
            r = st.number_input("无风险利率 (%)", value=2.5) / 100
        with col2:
            q = st.number_input("股息率 (%)", value=0.5) / 100
            sigma = st.number_input("波动率 (%)", value=30.0) / 100
        expiry = st.date_input("到期日", value=datetime.today())
        T = (expiry - datetime.today().date()).days / 365
    
    try:
        # stock = yf.Ticker(ticker)
        stock = fetch_price(ticker)
        S = stock[0]
        currency = stock[1]
    except:
        st.error("标的代码无效或数据不可用")
        return
    
    if st.button("计算分析"):
        price = american_option_price(S, K, T, r, q, sigma, option_type.lower())
        
        price_range = np.linspace(S*0.5, S*1.5, 100)
        premiums = [american_option_price(s, K, T, r, q, sigma, option_type.lower()) for s in price_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range, y=premiums, name='期权价格曲线'))
        fig.add_vline(x=S, line_dash="dash", line_color="green")
        
        st.metric(f"当前标的物价格 ({currency})", f"{S:.2f}", 
                 f"期权理论价格: {price:.2f} {currency}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("风险指标")
        cols = st.columns(4)
        with cols[0]:
            delta = (premiums[-1] - premiums[0])/(price_range[-1] - price_range[0])
            st.metric("Delta", f"{delta:.4f}")
        with cols[1]:
            gamma = ((premiums[-1] - premiums[-2]) - (premiums[1] - premiums[0]))/(price_range[1] - price_range[0])
            st.metric("Gamma", f"{gamma:.4f}")
        with cols[2]:
            theta = (american_option_price(S, K, max(T-0.01,0.001), r, q, sigma, option_type.lower()) - price)/0.01
            st.metric("Theta", f"{theta:.4f}")
        with cols[3]:
            vega = (american_option_price(S, K, T, r, q, sigma+0.01, option_type.lower()) - price)/0.01
            st.metric("Vega", f"{vega:.4f}")

if __name__ == "__main__":
    main()