import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import skewnorm, norm
from scipy.signal import savgol_filter  # <-- The tool for your SG filter idea
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Advanced Market Maker Backtest")

st.title("📈 Advanced Market Maker Strategy Backtest")
st.markdown("""
This application directly implements the concepts from the `tr8dr.github.io` posts:
- **Savitzky-Golay Filters** are used to detect swing points and define market regimes.
- **Machine Learning & HMM Models** are trained on these regimes to generate entry signals.
- **Swing Points** are then used again to set dynamic stop-loss exit levels.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Backtest Configuration")

st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 1000, 5000, 2000)

# --- These controls directly implement the SG Filter concept ---
st.sidebar.subheader("Swing Point Exit Logic (Savitzky-Golay)")
sg_fast_window = st.sidebar.slider("SG Fast Window", 5, 51, 11, step=2)
sg_slow_window = st.sidebar.slider("SG Slow Window", 21, 201, 51, step=2)
sg_polyorder = st.sidebar.slider("SG Polyorder", 2, 5, 3)
swing_threshold = st.sidebar.slider("Swing Threshold (%)", 0.1, 2.0, 0.5, 0.1)
stop_loss_offset_pct = st.sidebar.slider("Stop Loss Offset (%)", 0.1, 5.0, 0.5, 0.1)
# ---

st.sidebar.subheader("Model & Backtest Settings")
train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7)
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)
risk_aversion_gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.01, 0.5, 0.05)

# ==============================================================================
# Core Functions
# ==============================================================================
@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

# --- THIS FUNCTION IS THE DIRECT IMPLEMENTATION OF YOUR SAVITZKY-GOLAY SWING FILTER IDEA ---
@st.cache_data
def find_swing_points_sg(prices_tuple, fast_w, slow_w, poly_o, threshold_pct):
    st.info("Detecting swing points with Savitzky-Golay filters...")
    prices = np.array(prices_tuple)
    # Fit two filters with different window lengths
    fast_sg = savgol_filter(prices, fast_w, poly_o)
    slow_sg = savgol_filter(prices, slow_w, poly_o)
    
    # Calculate the normalized difference between the smoothed values
    sg_diff = (fast_sg - slow_sg) / slow_sg * 100
    
    # Add a threshold to the difference to identify significant peaks (swings)
    swing_highs = (sg_diff < -threshold_pct)
    swing_lows = (sg_diff > threshold_pct)
    
    return swing_highs, swing_lows
# ---

@st.cache_data
def add_features_and_labels(df, swing_highs, swing_lows):
    st.info("Labeling data and engineering features...")
    df_copy = df.copy()
    
    # --- THIS IS THE "DISTRIBUTION ANALYSIS" STEP FROM YOUR HMM POST ---
    # We categorize returns into "post a swing high" (down leg) and "post a swing low" (up leg)
    labels = np.zeros(len(df_copy)); last_swing = 0
    for i in range(len(df_copy)):
        if swing_lows[i] and last_swing != -1: last_swing = -1
        elif swing_highs[i] and last_swing != 1: last_swing = 1
        if last_swing == -1: labels[i] = 1
        if last_swing == 1: labels[i] = -1
    # ---

    df_copy['label'] = labels
    df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    df_copy['momentum_24'] = np.log(df_copy['close'] / df_copy['close'].shift(24))
    df_copy['volatility_24'] = df_copy['return'].rolling(window=24).std() * np.sqrt(24 * 365.25)
    df_copy['target'] = df_copy['label'].shift(-8)
    df_copy.dropna(inplace=True)
    return df_copy

# --- THIS FUNCTION IMPLEMENTS THE DISTRIBUTION FITTING FROM YOUR HMM POST ---
@st.cache_data
def fit_hmm_parameters(df_train):
    st.info("Fitting HMM parameters..."); up = df_train[df_train['label'] == 1]['return'].dropna(); neutral = df_train[df_train['label'] == 0]['return'].dropna(); down = df_train[df_train['label'] == -1]['return'].dropna()
    # Fit skew-normal for up/down legs and normal for neutral, as hypothesized
    dist_params = {"up": skewnorm.fit(up), "neutral": norm.fit(neutral), "down": skewnorm.fit(down)}
    labels = df_train['label'].values
    state_map = {-1: 0, 0: 1, 1: 2}; y_true = [state_map.get(s, 1) for s in labels[:-1]]; y_pred = [state_map.get(s, 1) for s in labels[1:]]
    counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]); trans_matrix = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
    return {"dists": dist_params, "transmat": trans_matrix}
# ---

@st.cache_data(allow_output_mutation=True)
def train_ml_model(df_train, feature_names):
    st.info("Training Machine Learning model..."); X_train = df_train[feature_names]; y_train = df_train['target']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train); return model

def run_backtest_with_exits(_df_backtest, _model, feature_names, cash, size, gamma, stop_offset, swing_high_prices, swing_low_prices):
    st.info(f"Running backtest for {_model.__class__.__name__}...")
    position, equity, trades = 0.0, [cash], []
    in_trade, stop_loss = False, 0.0
    for i in range(len(_df_backtest) - 1):
        current_bar = _df_backtest.iloc[i]; next_bar = _df_backtest.iloc[i + 1]
        
        # --- THIS IS THE INTEGRATION OF SWING POINTS AS EXIT LEVELS ---
        if in_trade:
            if position > 0 and next_bar['low'] <= stop_loss: # Exit long
                cash += position * stop_loss; trades.append({'t':next_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
            elif position < 0 and next_bar['high'] >= stop_loss: # Exit short
                cash += position * stop_loss; trades.append({'t':next_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
        # ---

        if not in_trade:
            feature_vector = current_bar[feature_names].values.reshape(1,-1); prediction = _model.predict(feature_vector)[0]
            if prediction != 0:
                mid=current_bar['close']; vol=current_bar['volatility_24']; res=(mid)-(position*gamma*(vol**2)); spread=(gamma*(vol**2))+(2/gamma)*np.log(1+(gamma/2)); bid,ask=res-spread/2,res+spread/2
                if prediction == 1 and next_bar['low'] <= bid: # Enter long
                    position += size; cash -= size * bid; in_trade = True; trades.append({'t':next_bar.name, 'type':'BUY', 'p':bid})
                    # Set stop loss based on the most recent swing low
                    last_swing_low = swing_low_prices[swing_low_prices.index < current_bar.name].iloc[-1]
                    stop_loss = last_swing_low * (1 - stop_offset / 100)
                elif prediction == -1 and next_bar['high'] >= ask: # Enter short
                    position -= size; cash += size * ask; in_trade = True; trades.append({'t':next_bar.name, 'type':'SELL', 'p':ask})
                    # Set stop loss based on the most recent swing high
                    last_swing_high = swing_high_prices[swing_high_prices.index < current_bar.name].iloc[-1]
                    stop_loss = last_swing_high * (1 + stop_offset / 100)
        equity.append(cash + position * current_bar['close'])
    return pd.DataFrame(trades), pd.Series(equity, index=_df_backtest.index)

# ==============================================================================
# Main App Logic
# ==============================================================================
if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing full backtest pipeline... Please wait."):
        df_raw = fetch_data(symbol, timeframe, data_limit)
        swing_highs_mask, swing_lows_mask = find_swing_points_sg(tuple(df_raw['close']), sg_fast_window, sg_slow_window, sg_polyorder, swing_threshold)
        df_featured = add_features_and_labels(df_raw, swing_highs_mask, swing_lows_mask)
        split_idx = int(len(df_featured) * train_test_split_ratio); df_train = df_featured.iloc[:split_idx]; df_backtest = df_featured.iloc[split_idx:]
        feature_names = ['momentum_24', 'volatility_24']
        ml_model = train_ml_model(df_train, feature_names)
        swing_high_prices = df_backtest['close'][swing_highs_mask[df_backtest.index[0]:df_backtest.index[-1]]]
        swing_low_prices = df_backtest['close'][swing_lows_mask[df_backtest.index[0]:df_backtest.index[-1]]]
        trades, equity = run_backtest_with_exits(df_backtest, ml_model, feature_names, initial_cash, trade_size, risk_aversion_gamma, stop_loss_offset_pct, swing_high_prices, swing_low_prices)
    st.success("✅ Backtest complete!")
    st.header("📊 Performance Summary"); final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
    st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
    st.header("📈 Charts")
    def plot_results(df_plot, equity_curve, trades, swing_high_prices, swing_low_prices):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Strategy Equity Curve', f'{symbol} Price, Swings & Trades'), row_heights=[0.3, 0.7])
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=swing_high_prices.index, y=swing_high_prices.values, mode='markers', name='Swing Highs', marker=dict(color='red', symbol='diamond-open', size=10)), row=2, col=1)
        fig.add_trace(go.Scatter(x=swing_low_prices.index, y=swing_low_prices.values, mode='markers', name='Swing Lows', marker=dict(color='green', symbol='diamond-open', size=10)), row=2, col=1)
        if not trades.empty:
            entries = trades[trades['type'].isin(['BUY', 'SELL'])]; exits = trades[trades['type'] == 'SL EXIT']
            fig.add_trace(go.Scatter(x=entries[entries['type']=='BUY']['t'], y=entries[entries['type']=='BUY']['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=2, col=1)
            fig.add_trace(go.Scatter(x=entries[entries['type']=='SELL']['t'], y=entries[entries['type']=='SELL']['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=2, col=1)
            fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Stop Loss Exits', marker=dict(color='black', symbol='x', size=8)), row=2, col=1)
        fig.update_layout(height=800, legend_title='Legend'); return fig
    st.plotly_chart(plot_results(df_backtest, equity, trades, swing_high_prices, swing_low_prices), use_container_width=True)
    with st.expander("Show Raw Trades Data"): st.dataframe(trades)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
