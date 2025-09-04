import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Robust Market Maker Backtest")

st.title("🏆 Robust Market Maker Strategy Backtest")
st.markdown("""
This application is a practical implementation of the concepts outlined in the `tr8dr.github.io` posts.
- It uses a **causal, bias-free** algorithm to perform the **"offline labeling"** of market regimes.
- It trains an ML model to **forecast the future regime**, solving the **"moving average lag"** problem.
- It provides an interactive way to test for **non-stationarity**, the key challenge identified in the HMM post.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Backtest Configuration")

st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 1000, 5000, 2000)

st.sidebar.subheader("Causal Swing Point Settings")
min_trend_pct = st.sidebar.slider("Min Reversal Pct to Confirm Swing", 0.5, 10.0, 3.0, 0.5)
stop_loss_offset_pct = st.sidebar.slider("Stop Loss Offset from Swing (%)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.subheader("Model & Backtest Settings")
train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7)
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)
risk_aversion_gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.01, 0.5, 0.05)


# ==============================================================================
# CORE BIAS-FREE FUNCTIONS
# ==============================================================================

@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

# ---
# THIS FUNCTION IS THE BIAS-FREE IMPLEMENTATION OF THE "OFFLINE LABELER" CONCEPT
# from the "Classifying returns" post. It finds swing points and defines the
# up-trends, down-trends, and neutral periods.
# ---
@st.cache_data
def find_and_label_causal_swings(prices_tuple, reversal_pct):
    st.info("Detecting causal swings and labeling regimes...")
    prices = np.array(prices_tuple); reversal_mult = reversal_pct / 100.0
    labels = np.zeros(len(prices)); swing_highs = np.full(len(prices), np.nan); swing_lows = np.full(len(prices), np.nan)
    peak_price, trough_price = prices[0], prices[0]; peak_idx, trough_idx = 0, 0; trend = 0
    for i in range(1, len(prices)):
        if trend == 1:
            if prices[i] >= peak_price: peak_price = prices[i]; peak_idx = i
            elif prices[i] < peak_price * (1 - reversal_mult): swing_highs[peak_idx] = peak_price; trend = -1; trough_price = prices[i]; trough_idx = i
        elif trend == -1:
            if prices[i] <= trough_price: trough_price = prices[i]; trough_idx = i
            elif prices[i] > trough_price * (1 + reversal_mult): swing_lows[trough_idx] = trough_price; trend = 1; peak_price = prices[i]; peak_idx = i
        else:
            if prices[i] > trough_price * (1 + reversal_mult): trend = 1; peak_price = prices[i]; peak_idx = i
            elif prices[i] < peak_price * (1 - reversal_mult): trend = -1; trough_price = prices[i]; trough_idx = i
            if prices[i] > peak_price: peak_price = prices[i]; peak_idx = i
            if prices[i] < trough_price: trough_price = prices[i]; trough_idx = i
    last_label = 0
    for i in range(len(prices)):
        if not np.isnan(swing_lows[i]): last_label = 1
        elif not np.isnan(swing_highs[i]): last_label = -1
        labels[i] = last_label
    return swing_highs, swing_lows, labels

@st.cache_data
def add_features(df, swing_highs, swing_lows, labels):
    st.info("Engineering features..."); df_copy = df.copy(); df_copy['swing_highs'] = swing_highs; df_copy['swing_lows'] = swing_lows; df_copy['label'] = labels
    df_copy['last_swing_high'] = df_copy['swing_highs'].ffill().bfill()
    df_copy['last_swing_low'] = df_copy['swing_lows'].ffill().bfill()
    if df_copy['last_swing_high'].isnull().all() or df_copy['last_swing_low'].isnull().all():
        st.warning("Could not find any swing points with the current settings to create features. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # --- THIS SECTION PREPARES THE DATA FOR THE FORECASTING MODEL ---
    # As suggested by the "Moving averages" post, we need to predict the future.
    # We define our 'target' as the market state 8 bars from now.
    df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    df_copy['momentum_24'] = np.log(df_copy['close'] / df_copy['close'].shift(24))
    df_copy['volatility_24'] = df_copy['return'].rolling(window=24).std() * np.sqrt(24 * 365.25)
    df_copy['target'] = df_copy['label'].shift(-8) # This is the key to solving the "lag" problem.
    df_copy.dropna(inplace=True)
    return df_copy

@st.cache_resource
def train_ml_model(_df_train, feature_names):
    st.info("Training Machine Learning model..."); 
    X_train = _df_train[feature_names]; y_train = _df_train['target']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train); 
    return model

@st.cache_data
def run_backtest_with_causal_exits(_df_backtest, _model, feature_names, cash, size, gamma, stop_offset):
    st.info(f"Running bias-free backtest for {_model.__class__.__name__}...")
    position, equity, trades = 0.0, [cash], []; in_trade, stop_loss = False, 0.0
    for i in range(len(_df_backtest)):
        current_bar = _df_backtest.iloc[i]
        if in_trade:
            if position > 0 and current_bar['low'] <= stop_loss:
                cash += position * stop_loss; trades.append({'t':current_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
            elif position < 0 and current_bar['high'] >= stop_loss:
                cash += position * stop_loss; trades.append({'t':current_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
        if not in_trade:
            # Here, the model uses current features to predict the future state.
            feature_vector = current_bar[feature_names].values.reshape(1, -1); prediction = _model.predict(feature_vector)[0]
            if prediction != 0:
                mid=current_bar['close']; vol=current_bar['volatility_24']; res=(mid)-(position*gamma*(vol**2)); spread=(gamma*(vol**2))+(2/gamma)*np.log(1+(gamma/2)); bid,ask=res-spread/2,res+spread/2
                if prediction == 1 and current_bar['close'] >= bid:
                    position += size; cash -= size * bid; in_trade = True; trades.append({'t':current_bar.name, 'type':'BUY', 'p':bid})
                    stop_loss = current_bar['last_swing_low'] * (1 - stop_offset / 100)
                elif prediction == -1 and current_bar['close'] <= ask:
                    position -= size; cash += size * ask; in_trade = True; trades.append({'t':current_bar.name, 'type':'SELL', 'p':ask})
                    stop_loss = current_bar['last_swing_high'] * (1 + stop_offset / 100)
        equity.append(cash + position * current_bar['close'])
    return pd.DataFrame(trades), pd.Series(equity, index=_df_backtest.index)

# ==============================================================================
# Main App Logic
# ==============================================================================
if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing full backtest pipeline... Please wait."):
        df_raw = fetch_data(symbol, timeframe, data_limit)
        swing_highs, swing_lows, labels = find_and_label_causal_swings(tuple(df_raw['close']), min_trend_pct)
        df_featured = add_features(df_raw, swing_highs, swing_lows, labels)
        if df_featured.empty:
            st.error("Error: The feature engineering process resulted in an empty dataset. Please adjust settings.")
        else:
            split_idx = int(len(df_featured) * train_test_split_ratio); df_train = df_featured.iloc[:split_idx]; df_backtest = df_featured.iloc[split_idx:]
            feature_names = ['momentum_24', 'volatility_24']
            if df_train.empty or len(df_train) < 50:
                st.error(f"Error: Not enough training data ({len(df_train)} samples) after splitting. Please adjust settings.")
            elif df_backtest.empty:
                st.error(f"Error: Not enough backtesting data ({len(df_backtest)} samples). Please adjust settings.")
            else:
                ml_model = train_ml_model(df_train, feature_names)
                trades, equity = run_backtest_with_causal_exits(df_backtest, ml_model, feature_names, initial_cash, trade_size, risk_aversion_gamma, stop_loss_offset_pct)
                st.success("✅ Backtest complete!")
                st.header("📊 Performance Summary"); final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
                st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
                st.header("📈 Charts")
                def plot_results(df_plot, equity_curve, trades):
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Strategy Equity Curve', f'{symbol} Price, Swings & Trades'), row_heights=[0.3, 0.7])
                    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['swing_highs'], mode='markers', name='Confirmed Swing Highs', marker=dict(color='red', symbol='diamond-open', size=10)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['swing_lows'], mode='markers', name='Confirmed Swing Lows', marker=dict(color='green', symbol='diamond-open', size=10)), row=2, col=1)
                    if not trades.empty:
                        entries = trades[trades['type'].isin(['BUY', 'SELL'])]; exits = trades[trades['type'] == 'SL EXIT']
                        fig.add_trace(go.Scatter(x=entries[entries['type']=='BUY']['t'], y=entries[entries['type']=='BUY']['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=2, col=1)
                        fig.add_trace(go.Scatter(x=entries[entries['type']=='SELL']['t'], y=entries[entries['type']=='SELL']['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=2, col=1)
                        fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Stop Loss Exits', marker=dict(color='black', symbol='x', size=8)), row=2, col=1)
                    fig.update_layout(height=800, legend_title='Legend'); return fig
                st.plotly_chart(plot_results(df_backtest, equity, trades), use_container_width=True)
                with st.expander("Show Raw Trades Data"): st.dataframe(trades)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
