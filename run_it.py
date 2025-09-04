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
st.set_page_config(layout="wide", page_title="Hawkes Process BSI Backtest")

st.title("🔥 Hawkes Process Buy/Sell Imbalance (BSI) Strategy Backtest")
st.markdown("""
This application backtests a momentum strategy based on the **HawkesBSI** indicator, inspired by the research at `tr8dr.github.io`.
- It models the intensity of buy/sell pressure as a self-exciting process.
- **Entry signals** are triggered when the BSI crosses a momentum threshold.
- **Exit signals** are triggered when the BSI reverts to zero, indicating momentum has faded.
- This approach is causal, avoids lookahead bias, and is designed for real-time analysis.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Strategy Configuration")

st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 500, 5000, 1500)

st.sidebar.subheader("Hawkes Process BSI Settings")
kappa = st.sidebar.slider("Decay Factor (Kappa)", 0.01, 1.0, 0.1, 0.01)
entry_threshold = st.sidebar.slider("Entry Threshold (BSI Value)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.subheader("Backtest Settings")
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)


# ==============================================================================
# CORE HAWKES BSI IMPLEMENTATION
# ==============================================================================

@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

# --- THIS IS THE DIRECT IMPLEMENTATION OF THE HAWKES BSI FROM THE RESEARCH ---
@st.cache_data
def calculate_hawkes_bsi(_df, kappa_decay):
    st.info("Simulating buy/sell volume and calculating Hawkes BSI...")
    df_copy = _df.copy()
    
    # Step 1: Simulate buy and sell volume (heuristic)
    price_change = df_copy['close'] - df_copy['open']
    # If price went up, assign 75% of volume to buys. If down, 25%.
    buy_ratio = np.where(price_change > 0, 0.75, 0.25)
    df_copy['buyvolume'] = df_copy['volume'] * buy_ratio
    df_copy['sellvolume'] = df_copy['volume'] * (1 - buy_ratio)
    
    # Step 2: Calculate signed volume imbalance (N(i) in the formula)
    imbalance = df_copy['buyvolume'] - df_copy['sellvolume']
    
    # Step 3: Apply the self-exciting Hawkes process formula iteratively
    bsi = np.zeros(len(df_copy))
    for i in range(1, len(df_copy)):
        # H(t) = (Previous H(t-1) * e^(-kappa)) + N(i)
        # We assume delta_t is 1 bar, so e^(-kappa * delta_t) is just e^(-kappa)
        bsi[i] = bsi[i-1] * np.exp(-kappa_decay) + imbalance.iloc[i]
        
    df_copy['bsi'] = bsi
    df_copy.dropna(inplace=True)
    return df_copy
# ---

@st.cache_data
def run_hawkes_backtest(_df, cash, size, entry_thresh):
    st.info("Running Hawkes BSI backtest...")
    position, equity, trades = 0.0, [cash], []
    
    for i in range(1, len(_df)): # Start from 1 to have a previous bar
        current_bar = _df.iloc[i]
        prev_bar = _df.iloc[i-1]
        
        bsi_current = current_bar['bsi']
        bsi_prev = prev_bar['bsi']
        
        # --- EXIT LOGIC: Exit when BSI crosses back through zero ---
        if position > 0 and bsi_prev > 0 and bsi_current <= 0:
            cash += position * current_bar['open']
            trades.append({'t': current_bar.name, 'type': 'EXIT LONG', 'p': current_bar['open']})
            position = 0
        elif position < 0 and bsi_prev < 0 and bsi_current >= 0:
            cash += position * current_bar['open']
            trades.append({'t': current_bar.name, 'type': 'EXIT SHORT', 'p': current_bar['open']})
            position = 0

        # --- ENTRY LOGIC: Enter when BSI crosses the threshold ---
        if position == 0:
            if bsi_prev < entry_thresh and bsi_current >= entry_thresh:
                position += size
                cash -= size * current_bar['close']
                trades.append({'t': current_bar.name, 'type': 'BUY', 'p': current_bar['close']})
            elif bsi_prev > -entry_thresh and bsi_current <= -entry_thresh:
                position -= size
                cash += size * current_bar['close']
                trades.append({'t': current_bar.name, 'type': 'SELL', 'p': current_bar['close']})
            
        equity.append(cash + position * current_bar['close'])
        
    return pd.DataFrame(trades), pd.Series(equity[1:], index=_df.index[1:])

# ==============================================================================
# Main App Logic
# ==============================================================================

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing backtest pipeline... Please wait."):
        df_raw = fetch_data(symbol, timeframe, data_limit)
        df_bsi = calculate_hawkes_bsi(df_raw, kappa)
        
        if df_bsi.empty:
            st.error("Error: The dataset is empty after feature calculation. Please increase the 'Number of Data Bars'.")
        else:
            trades, equity = run_hawkes_backtest(df_bsi, initial_cash, trade_size, entry_threshold)

            st.success("✅ Backtest complete!")
            st.header("📊 Performance Summary")
            final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
            st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
            
            st.header("📈 Charts")
            def plot_results(df_plot, equity_curve, trades):
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    subplot_titles=('Strategy Equity Curve', 'Hawkes BSI Indicator', f'{symbol} Price & Trades'),
                                    row_heights=[0.25, 0.25, 0.5])
                
                # Plot 1: Equity
                fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
                
                # Plot 2: Hawkes BSI
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bsi'], mode='lines', name='BSI', line=dict(color='orange')), row=2, col=1)
                fig.add_hline(y=entry_threshold, line_width=2, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Buy Threshold")
                fig.add_hline(y=-entry_threshold, line_width=2, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Sell Threshold")
                fig.add_hline(y=0, line_width=1, line_color="grey", row=2, col=1)

                # Plot 3: Price and Trades
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=3, col=1)
                
                if not trades.empty:
                    buys = trades[trades['type'] == 'BUY']; sells = trades[trades['type'] == 'SELL']; exits = trades[trades['type'].str.contains('EXIT')]
                    fig.add_trace(go.Scatter(x=buys['t'], y=buys['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=sells['t'], y=sells['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Exits', marker=dict(color='black', symbol='x', size=8)), row=3, col=1)
                
                fig.update_layout(height=900, legend_title='Legend'); return fig
            
            st.plotly_chart(plot_results(df_bsi, equity, trades), use_container_width=True)
            with st.expander("Show Raw Trades Data"): st.dataframe(trades)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
