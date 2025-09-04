import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Robust Strategy Backtest")

st.title("🏆 Robust Trend-Following Strategy Backtest")
st.markdown("""
This application is a practical implementation of the concepts outlined in the `tr8dr.github.io` posts.
- It uses a **robust, causal Price Channel** method as the **"Offline Labeler"** to classify market regimes.
- The backtest then acts as an **"Online Classifier"**, trading directly on these labels.
- The interactive sliders allow you to test for **non-stationarity**, the key challenge identified in the research.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Strategy Configuration")

st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 500, 5000, 1500)

# --- These controls directly implement the "Labeling Momentum & Trends" concept ---
st.sidebar.subheader("Trend Definition (Price Channel)")
# The "Trend Channel Window" is analogous to the 'Tinactive' parameter in AmplitudeBasedLabeler
trend_window = st.sidebar.slider("Trend Channel Window (bars)", 10, 200, 24)
# The "minamp" is implicitly handled by the quality of the trend signals generated.
# ---

st.sidebar.subheader("Backtest Settings")
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)


# ==============================================================================
# CORE ROBUST FUNCTIONS
# ==============================================================================

@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

# ---
# THIS FUNCTION IS THE ROBUST IMPLEMENTATION OF THE "OFFLINE LABELER" CONCEPT.
# It classifies historical data into up-trends (+1), down-trends (-1), and neutral (0)
# based on a rolling price channel, as inspired by the AmplitudeBasedLabeler.
# ---
@st.cache_data
def label_regimes_by_price_channel(df, window):
    st.info("Labeling market regimes using Price Channel...")
    df_copy = df.copy()
    
    df_copy['channel_high'] = df_copy['high'].rolling(window=window).max()
    df_copy['channel_low'] = df_copy['low'].rolling(window=window).min()
    
    df_copy['label'] = 0
    df_copy.loc[df_copy['close'] >= df_copy['channel_high'], 'label'] = 1
    df_copy.loc[df_copy['close'] <= df_copy['channel_low'], 'label'] = -1
    
    df_copy.dropna(inplace=True)
    return df_copy
# ---

@st.cache_data
def run_simple_backtest(_df, cash, size):
    st.info("Running simplified backtest...")
    position, equity, trades = 0.0, [cash], []
    
    # --- THIS LOOP ACTS AS THE "ONLINE CLASSIFIER" ---
    # It iterates through time, making decisions based only on the
    # most recent label, simulating a live environment.
    for i in range(len(_df)):
        current_bar = _df.iloc[i]
        label = current_bar['label']
        
        # Exit Logic
        if position > 0 and label != 1:
            cash += position * current_bar['open']
            trades.append({'t': current_bar.name, 'type': 'EXIT LONG', 'p': current_bar['open']})
            position = 0
        elif position < 0 and label != -1:
            cash += position * current_bar['open']
            trades.append({'t': current_bar.name, 'type': 'EXIT SHORT', 'p': current_bar['open']})
            position = 0

        # Entry Logic
        if label == 1 and position == 0:
            position += size
            cash -= size * current_bar['close']
            trades.append({'t': current_bar.name, 'type': 'BUY', 'p': current_bar['close']})
        elif label == -1 and position == 0:
            position -= size
            cash += size * current_bar['close']
            trades.append({'t': current_bar.name, 'type': 'SELL', 'p': current_bar['close']})
            
        equity.append(cash + position * current_bar['close'])
        
    return pd.DataFrame(trades), pd.Series(equity[1:], index=_df.index)

# ==============================================================================
# Main App Logic
# ==============================================================================

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing backtest pipeline... Please wait."):
        df_raw = fetch_data(symbol, timeframe, data_limit)
        df_labeled = label_regimes_by_price_channel(df_raw, trend_window)

        if df_labeled.empty:
            st.error("Error: The dataset is empty after the initial lookback period. Please increase the 'Number of Data Bars'.")
        else:
            trades, equity = run_simple_backtest(df_labeled, initial_cash, trade_size)

            st.success("✅ Backtest complete!")
            st.header("📊 Performance Summary")
            final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
            st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
            
            st.header("📈 Charts")
            def plot_results(df_plot, equity_curve, trades):
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Strategy Equity Curve', f'{symbol} Price, Trend Channels & Trades'), row_heights=[0.3, 0.7])
                fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['channel_high'], mode='lines', name='Trend Channel High', line=dict(color='lightgrey', dash='dash')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['channel_low'], mode='lines', name='Trend Channel Low', line=dict(color='lightgrey', dash='dash')), row=2, col=1)
                if not trades.empty:
                    buys = trades[trades['type'] == 'BUY']; sells = trades[trades['type'] == 'SELL']; exits = trades[trades['type'].str.contains('EXIT')]
                    fig.add_trace(go.Scatter(x=buys['t'], y=buys['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=sells['t'], y=sells['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Exits', marker=dict(color='black', symbol='x', size=8)), row=2, col=1)
                fig.update_layout(height=800, legend_title='Legend'); return fig
            
            st.plotly_chart(plot_results(df_labeled, equity, trades), use_container_width=True)
            with st.expander("Show Raw Trades Data"): st.dataframe(trades)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
