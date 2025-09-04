import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Hawkes Momentum Backtest")

st.title("🔥 Hawkes Process Momentum Strategy Backtest")
st.markdown("""
This application backtests and compares two momentum indicators inspired by the research at `tr8dr.github.io`:
1.  **HawkesBSI:** A simple model using a heuristic for buy/sell volume.
2.  **HawkesBVC:** A more advanced model using **Bulk Volume Classification** to statistically estimate buy/sell pressure from the return distribution.

Use the sidebar to choose your indicator and configure the backtest.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Strategy Configuration")

st.sidebar.subheader("Indicator & Data Settings")
# --- NEW: Allow user to choose the indicator ---
indicator_type = st.sidebar.selectbox("Indicator Type", ["HawkesBVC", "HawkesBSI"])
# ---
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 500, 5000, 1500)

st.sidebar.subheader("Hawkes Process Settings")
kappa = st.sidebar.slider("Decay Factor (Kappa)", 0.01, 1.0, 0.1, 0.01)
# --- NEW: Conditional widget for BVC ---
if indicator_type == "HawkesBVC":
    volatility_window = st.sidebar.slider("BVC Volatility Window", 10, 100, 30)
# ---
entry_threshold = st.sidebar.slider("Entry Threshold", 0.1, 5.0, 1.0, 0.1)

st.sidebar.subheader("Backtest Settings")
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)


# ==============================================================================
# CORE INDICATOR IMPLEMENTATIONS
# ==============================================================================

@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

@st.cache_data
def calculate_hawkes_bsi(_df, kappa_decay):
    st.info("Simulating buy/sell volume and calculating Hawkes BSI..."); df_copy = _df.copy()
    price_change = df_copy['close'] - df_copy['open']
    buy_ratio = np.where(price_change > 0, 0.75, 0.25)
    df_copy['buyvolume'] = df_copy['volume'] * buy_ratio; df_copy['sellvolume'] = df_copy['volume'] * (1 - buy_ratio)
    imbalance = df_copy['buyvolume'] - df_copy['sellvolume']
    bsi = np.zeros(len(df_copy))
    for i in range(1, len(df_copy)): bsi[i] = bsi[i-1] * np.exp(-kappa_decay) + imbalance.iloc[i]
    df_copy['indicator'] = bsi
    df_copy.dropna(inplace=True)
    return df_copy

# --- THIS IS THE NEW, DIRECT IMPLEMENTATION OF THE HAWKES BVC FROM THE RESEARCH ---
@st.cache_data
def calculate_hawkes_bvc(_df, vol_window, kappa_decay):
    st.info("Classifying volume with BVC and calculating Hawkes BVC...")
    df_copy = _df.copy()
    
    # Step 1: Calculate returns and rolling volatility (sigma_t)
    df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    df_copy['volatility'] = df_copy['return'].rolling(window=vol_window).std()
    
    # Step 2: Calculate the normalized return (z-score)
    df_copy['z_score'] = df_copy['return'] / (df_copy['volatility'] + 1e-9)
    
    # Step 3: Calculate the CDF for the bulk volume classification
    # We assume returns are normally distributed, so we use the normal CDF
    df_copy['cdf'] = norm.cdf(df_copy['z_score'])
    
    # Step 4: Calculate signed volume (N(i) in the formula)
    # signed_volume = volume * (2 * cdf - 1)
    df_copy['signed_volume'] = df_copy['volume'] * (2 * df_copy['cdf'] - 1)
    
    # Step 5: Apply the self-exciting Hawkes process formula iteratively
    bvc = np.zeros(len(df_copy))
    for i in range(1, len(df_copy)):
        bvc[i] = bvc[i-1] * np.exp(-kappa_decay) + df_copy['signed_volume'].iloc[i]
        
    df_copy['indicator'] = bvc
    df_copy.dropna(inplace=True)
    return df_copy
# ---

@st.cache_data
def run_hawkes_backtest(_df, cash, size, entry_thresh, indicator_col='indicator'):
    st.info(f"Running backtest on {indicator_col.upper()} signal...")
    position, equity, trades = 0.0, [cash], []
    for i in range(1, len(_df)):
        current_bar, prev_bar = _df.iloc[i], _df.iloc[i-1]
        signal_current, signal_prev = current_bar[indicator_col], prev_bar[indicator_col]
        if position > 0 and signal_prev > 0 and signal_current <= 0:
            cash += position * current_bar['open']; trades.append({'t': current_bar.name, 'type': 'EXIT LONG', 'p': current_bar['open']}); position = 0
        elif position < 0 and signal_prev < 0 and signal_current >= 0:
            cash += position * current_bar['open']; trades.append({'t': current_bar.name, 'type': 'EXIT SHORT', 'p': current_bar['open']}); position = 0
        if position == 0:
            if signal_prev < entry_thresh and signal_current >= entry_thresh:
                position += size; cash -= size * current_bar['close']; trades.append({'t': current_bar.name, 'type': 'BUY', 'p': current_bar['close']})
            elif signal_prev > -entry_thresh and signal_current <= -entry_thresh:
                position -= size; cash += size * current_bar['close']; trades.append({'t': current_bar.name, 'type': 'SELL', 'p': current_bar['close']})
        equity.append(cash + position * current_bar['close'])
    return pd.DataFrame(trades), pd.Series(equity[1:], index=_df.index[1:])

# ==============================================================================
# Main App Logic
# ==============================================================================

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing backtest pipeline... Please wait."):
        df_raw = fetch_data(symbol, timeframe, data_limit)
        
        # --- MODIFIED: Choose which indicator to calculate and run ---
        if indicator_type == "HawkesBSI":
            df_indicator = calculate_hawkes_bsi(df_raw, kappa)
        elif indicator_type == "HawkesBVC":
            df_indicator = calculate_hawkes_bvc(df_raw, volatility_window, kappa)
        # ---

        if df_indicator.empty:
            st.error("Error: The dataset is empty after feature calculation. Please increase the 'Number of Data Bars'.")
        else:
            trades, equity = run_hawkes_backtest(df_indicator, initial_cash, trade_size, entry_threshold)

            st.success("✅ Backtest complete!")
            st.header("📊 Performance Summary")
            st.subheader(f"Strategy: {indicator_type}")
            final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
            st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
            
            st.header("📈 Charts")
            def plot_results(df_plot, equity_curve, trades):
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    subplot_titles=('Strategy Equity Curve', f'{indicator_type} Indicator', f'{symbol} Price & Trades'),
                                    row_heights=[0.25, 0.25, 0.5])
                fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['indicator'], mode='lines', name=indicator_type, line=dict(color='orange')), row=2, col=1)
                fig.add_hline(y=entry_threshold, line_width=2, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Buy Threshold")
                fig.add_hline(y=-entry_threshold, line_width=2, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Sell Threshold")
                fig.add_hline(y=0, line_width=1, line_color="grey", row=2, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=3, col=1)
                if not trades.empty:
                    buys = trades[trades['type'] == 'BUY']; sells = trades[trades['type'] == 'SELL']; exits = trades[trades['type'].str.contains('EXIT')]
                    fig.add_trace(go.Scatter(x=buys['t'], y=buys['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=sells['t'], y=sells['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Exits', marker=dict(color='black', symbol='x', size=8)), row=3, col=1)
                fig.update_layout(height=900, legend_title='Legend'); return fig
            
            st.plotly_chart(plot_results(df_indicator, equity, trades), use_container_width=True)
            with st.expander("Show Raw Trades Data"): st.dataframe(trades)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
