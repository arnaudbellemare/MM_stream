import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import pywt # For Wavelet Denoising
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Quantitative Research Dashboard")

st.title(" QUANTITATIVE TRADING RESEARCH DASHBOARD")
st.markdown("An interactive dashboard for backtesting strategies and analyzing advanced market signals.")

# ==============================================================================
# Helper Functions (used across tabs)
# ==============================================================================
@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'timestamp': 'stamp'}, inplace=True) # Add stamp for matplotlib
    return df

# EMA function needed for Wavelet chart
def ema(data, window):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

# ==============================================================================
# TAB STRUCTURE
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["🏆 Strategy Backtester", "🔬 SG Swing Analysis", "🌊 Wavelet Auto-Labeling"])


# ==============================================================================
# TAB 1: STRATEGY BACKTESTER
# ==============================================================================
with tab1:
    st.header("Bias-Free Strategy Backtest")
    
    st.sidebar.header("⚙️ Backtester Configuration")
    st.sidebar.subheader("Data Settings (Backtester)")
    symbol_bt = st.sidebar.text_input("Symbol", "BTC/USDT", key="bt_symbol")
    timeframe_bt = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1, key="bt_tf")
    data_limit_bt = st.sidebar.slider("Number of Data Bars", 1000, 5000, 2000, key="bt_limit")

    st.sidebar.subheader("Causal Swing Point Settings")
    min_trend_pct = st.sidebar.slider("Min Reversal Pct to Confirm Swing", 0.5, 10.0, 3.0, 0.5, key="bt_reversal")
    stop_loss_offset_pct = st.sidebar.slider("Stop Loss Offset from Swing (%)", 0.1, 5.0, 1.0, 0.1, key="bt_sl")

    st.sidebar.subheader("Model & Backtest Settings")
    train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7, key="bt_split")
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0, key="bt_cash")
    trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1, key="bt_size")
    risk_aversion_gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.01, 0.5, 0.05, key="bt_gamma")

    @st.cache_data
    def find_and_label_causal_swings(prices_tuple, reversal_pct):
        prices = np.array(prices_tuple); reversal_mult = reversal_pct / 100.0; labels = np.zeros(len(prices)); swing_highs = np.full(len(prices), np.nan); swing_lows = np.full(len(prices), np.nan)
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
        df_copy = df.copy(); df_copy['swing_highs'] = swing_highs; df_copy['swing_lows'] = swing_lows; df_copy['label'] = labels; df_copy['last_swing_high'] = df_copy['swing_highs'].ffill().bfill(); df_copy['last_swing_low'] = df_copy['swing_lows'].ffill().bfill()
        if df_copy['last_swing_high'].isnull().all() or df_copy['last_swing_low'].isnull().all(): return pd.DataFrame()
        df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1)); df_copy['momentum_24'] = np.log(df_copy['close'] / df_copy['close'].shift(24)); df_copy['volatility_24'] = df_copy['return'].rolling(window=24).std() * np.sqrt(24 * 365.25)
        df_copy['target'] = df_copy['label'].shift(-8); df_copy.dropna(inplace=True)
        return df_copy

    @st.cache_resource
    def train_ml_model(_df_train, feature_names):
        X_train = _df_train[feature_names]; y_train = _df_train['target']; model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1); model.fit(X_train, y_train); return model

    @st.cache_data
    def run_backtest_with_causal_exits(_df_backtest, _model, feature_names, cash, size, gamma, stop_offset):
        position, equity, trades = 0.0, [cash], []; in_trade, stop_loss = False, 0.0
        for i in range(len(_df_backtest)):
            current_bar = _df_backtest.iloc[i]
            if in_trade:
                if position > 0 and current_bar['low'] <= stop_loss: cash += position * stop_loss; trades.append({'t':current_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
                elif position < 0 and current_bar['high'] >= stop_loss: cash += position * stop_loss; trades.append({'t':current_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
            if not in_trade:
                feature_vector = current_bar[feature_names].values.reshape(1, -1); prediction = _model.predict(feature_vector)[0]
                if prediction != 0:
                    mid=current_bar['close']; vol=current_bar['volatility_24']; res=(mid)-(position*gamma*(vol**2)); spread=(gamma*(vol**2))+(2/gamma)*np.log(1+(gamma/2)); bid,ask=res-spread/2,res+spread/2
                    if prediction == 1 and current_bar['close'] >= bid: position += size; cash -= size * bid; in_trade = True; trades.append({'t':current_bar.name, 'type':'BUY', 'p':bid}); stop_loss = current_bar['last_swing_low'] * (1 - stop_offset / 100)
                    elif prediction == -1 and current_bar['close'] <= ask: position -= size; cash += size * ask; in_trade = True; trades.append({'t':current_bar.name, 'type':'SELL', 'p':ask}); stop_loss = current_bar['last_swing_high'] * (1 + stop_offset / 100)
            equity.append(cash + position * current_bar['close'])
        return pd.DataFrame(trades), pd.Series(equity, index=_df_backtest.index)

    if st.sidebar.button("🚀 Run Backtest", key="run_bt"):
        with st.spinner("Executing backtest pipeline... Please wait."):
            df_raw_bt = fetch_data(symbol_bt, timeframe_bt, data_limit_bt)
            swing_highs_bt, swing_lows_bt, labels_bt = find_and_label_causal_swings(tuple(df_raw_bt['close']), min_trend_pct)
            df_featured_bt = add_features(df_raw_bt, swing_highs_bt, swing_lows_bt, labels_bt)
            if df_featured_bt.empty:
                st.error("Error: The feature engineering resulted in an empty dataset. Please adjust settings.")
            else:
                split_idx = int(len(df_featured_bt) * train_test_split_ratio); df_train_bt = df_featured_bt.iloc[:split_idx]; df_backtest_bt = df_featured_bt.iloc[split_idx:]
                feature_names = ['momentum_24', 'volatility_24']
                if df_train_bt.empty or len(df_train_bt) < 50: st.error("Error: Not enough training data.")
                elif df_backtest_bt.empty: st.error("Error: Not enough backtesting data.")
                else:
                    ml_model_bt = train_ml_model(df_train_bt, feature_names)
                    trades_bt, equity_bt = run_backtest_with_causal_exits(df_backtest_bt, ml_model_bt, feature_names, initial_cash, trade_size, risk_aversion_gamma, stop_loss_offset_pct)
                    st.success("✅ Backtest complete!")
                    final_equity = equity_bt.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
                    st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades", len(trades_bt))
                    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Strategy Equity Curve', f'{symbol_bt} Price, Swings & Trades'), row_heights=[0.3, 0.7])
                    fig_bt.add_trace(go.Scatter(x=equity_bt.index, y=equity_bt, mode='lines', name='Equity'), row=1, col=1)
                    fig_bt.add_trace(go.Scatter(x=df_backtest_bt.index, y=df_backtest_bt['close'], mode='lines', name=f'{symbol_bt} Price', line=dict(color='blue')), row=2, col=1)
                    fig_bt.add_trace(go.Scatter(x=df_backtest_bt.index, y=df_backtest_bt['swing_highs'], mode='markers', name='Confirmed Highs', marker=dict(color='red', symbol='diamond-open', size=10)), row=2, col=1)
                    fig_bt.add_trace(go.Scatter(x=df_backtest_bt.index, y=df_backtest_bt['swing_lows'], mode='markers', name='Confirmed Lows', marker=dict(color='green', symbol='diamond-open', size=10)), row=2, col=1)
                    if not trades_bt.empty:
                        entries = trades_bt[trades_bt['type'].isin(['BUY', 'SELL'])]; exits = trades_bt[trades_bt['type'] == 'SL EXIT']
                        fig_bt.add_trace(go.Scatter(x=entries[entries['type']=='BUY']['t'], y=entries[entries['type']=='BUY']['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=2, col=1)
                        fig_bt.add_trace(go.Scatter(x=entries[entries['type']=='SELL']['t'], y=entries[entries['type']=='SELL']['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=2, col=1)
                        fig_bt.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Exits', marker=dict(color='black', symbol='x', size=8)), row=2, col=1)
                    fig_bt.update_layout(height=800); st.plotly_chart(fig_bt, use_container_width=True)
                    with st.expander("Show Raw Trades Data"): st.dataframe(trades_bt)
    else:
        st.info("Adjust the backtester parameters in the sidebar and click 'Run Backtest'.")

# ==============================================================================
# TAB 2: SG SWING ANALYSIS
# ==============================================================================
with tab2:
    st.header("Savitzky-Golay Swing Point Detection with SpanB Overlay")
    st.markdown("This is an **offline analysis tool** for visualizing swing points, as discussed in the `tr8dr.github.io` research. Note that this method has a **lookahead bias** and is **not** used in the live backtester.")

    st.sidebar.subheader("SG Swing Analysis Settings")
    symbol_sg = st.sidebar.text_input("Symbol", "BTC/USDT", key="sg_symbol")
    window_short_sg = st.sidebar.slider("SG Fast Window", 5, 51, 33, step=2, key="sg_fast")
    window_long_sg = st.sidebar.slider("SG Slow Window", 101, 301, 257, step=2, key="sg_slow")
    polyorder_sg = st.sidebar.slider("SG Polyorder", 2, 5, 3, key="sg_poly")
    span_window_sg = st.sidebar.slider("SpanB Window", 20, 100, 52, key="sg_span")

    if st.sidebar.button("🔬 Run SG Analysis", key="run_sg"):
        df_sg = fetch_data(symbol_sg, '1h', 1000)
        if len(df_sg) < window_long_sg:
            st.error("Not enough data for the selected SG window length.")
        else:
            close_prices = df_sg['close'].values
            smoothed_short = savgol_filter(close_prices, window_length=window_short_sg, polyorder=polyorder_sg)
            smoothed_long = savgol_filter(close_prices, window_length=window_long_sg, polyorder=polyorder_sg)
            diff_signal = smoothed_short - smoothed_long
            threshold = np.std(diff_signal)
            swing_points = [i for i in range(1, len(diff_signal)-1) if np.abs(diff_signal[i])>threshold and np.abs(diff_signal[i])>np.abs(diff_signal[i-1]) and np.abs(diff_signal[i])>np.abs(diff_signal[i+1])]
            df_sg['spanB'] = (df_sg['high'].rolling(window=span_window_sg).max() + df_sg['low'].rolling(window=span_window_sg).min()) / 2
            
            # Plotting with Matplotlib
            fig_sg, ax_sg = plt.subplots(figsize=(15, 7))
            segments = np.array([mdates.date2num(df_sg.index), close_prices]).T.reshape(-1, 1, 2)
            segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
            lc = LineCollection(segments, cmap='bwr', norm=plt.Normalize(vmin=-np.abs(diff_signal).max(), vmax=np.abs(diff_signal).max()))
            lc.set_array(diff_signal)
            ax_sg.add_collection(lc)
            ax_sg.plot(df_sg.index, smoothed_short, label=f"Savgol Short", color='cyan', linestyle='--', lw=1)
            ax_sg.plot(df_sg.index, smoothed_long, label=f"Savgol Long", color='orange', linestyle='--', lw=1)
            ax_sg.scatter(df_sg.index[swing_points], close_prices[swing_points], color='lime', s=50, label="Swing Points", zorder=5)
            ax_sg.plot(df_sg.index, df_sg['spanB'], label="SpanB", color='purple', lw=2, linestyle='-.')
            ax_sg.set_xlim(df_sg.index.min(), df_sg.index.max()); ax_sg.set_ylim(close_prices.min()*0.98, close_prices.max()*1.02)
            ax_sg.legend(); ax_sg.set_title("SG Swing Point Analysis"); plt.colorbar(lc, ax=ax_sg, label="SG Filter Difference")
            st.pyplot(fig_sg)

# ==============================================================================
# TAB 3: WAVELET AUTO-LABELING
# ==============================================================================
with tab3:
    st.header("Wavelet Auto-Labeling & Performance Metrics")
    st.markdown("This section implements the advanced labeling technique using wavelet denoising and evaluates its performance against a simple ground truth.")

    st.sidebar.subheader("Wavelet Labeling Settings")
    symbol_wl = st.sidebar.text_input("Symbol", "BTC/USDT", key="wl_symbol")
    ema_window_wl = st.sidebar.slider("EMA Window for Chart", 5, 50, 10, key="wl_ema")
    threshold_type_wl = st.sidebar.radio("Threshold Type", ("Volatility-based", "Constant"), key="wl_thresh_type")
    constant_w_wl = st.sidebar.number_input("Constant Threshold (w)", value=0.01, step=0.001, format="%.4f", key="wl_const_w")

    def auto_labeling(data_list, timestamp_list, w):
        labels = np.zeros(len(data_list)); FP = data_list[0]; x_H = data_list[0]; HT = timestamp_list[0]; x_L = data_list[0]; LT = timestamp_list[0]; Cid = 0; FP_N = 0
        for i in range(len(data_list)):
            if data_list[i] > FP + data_list[0] * w: x_H = data_list[i]; HT = timestamp_list[i]; FP_N = i; Cid = 1; break
            if data_list[i] < FP - data_list[0] * w: x_L = data_list[i]; LT = timestamp_list[i]; FP_N = i; Cid = -1; break
        for i in range(FP_N, len(data_list)):
            if Cid > 0:
                if data_list[i] > x_H: x_H = data_list[i]; HT = timestamp_list[i]
                if data_list[i] < x_H - x_H * w and LT < HT:
                    for j in range(len(data_list)):
                        if timestamp_list[j] > LT and timestamp_list[j] <= HT: labels[j] = 1
                    x_L = data_list[i]; LT = timestamp_list[i]; Cid = -1
            elif Cid < 0:
                if data_list[i] < x_L: x_L = data_list[i]; LT = timestamp_list[i]
                if data_list[i] > x_L + x_L * w and HT <= LT:
                    for j in range(len(data_list)):
                        if timestamp_list[j] > HT and timestamp_list[j] <= LT: labels[j] = -1
                    x_H = data_list[i]; HT = timestamp_list[i]; Cid = 1
        labels[0] = labels[1] if len(labels) > 1 else Cid; labels = np.where(labels == 0, Cid, labels)
        return labels

    def rogers_satchell_volatility(data):
        log_ho = np.log(data["high"] / data["open"]); log_lo = np.log(data["low"] / data["open"]); log_co = np.log(data["close"] / data["open"])
        return np.sqrt(np.mean(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)))

    if st.sidebar.button("🌊 Run Wavelet Analysis", key="run_wl"):
        df_wl = fetch_data(symbol_wl, '1h', 1000)
        data_train = df_wl["close"].values; timestamps_train = df_wl.index.values
        
        st.info("Denoising data with Wavelets...")
        coeffs = pywt.wavedec(data_train, 'db4', level=4)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(data_train)))
        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
        data_train_denoised = pywt.waverec(coeffs_thresh, 'db4')
        
        if threshold_type_wl == "Volatility-based":
            w_used = rogers_satchell_volatility(df_wl)
            st.write(f"**Volatility-based threshold (w):** `{w_used:.4f}`")
        else:
            w_used = constant_w_wl
            st.write(f"**Using constant threshold (w):** `{w_used:.4f}`")

        st.info("Auto-labeling denoised data...")
        labels_wavelet = auto_labeling(data_train_denoised, timestamps_train, w_used)
        df_wl['label'] = labels_wavelet

        # Evaluate
        gt_labels = np.sign(df_wl['close'].shift(-1) - df_wl['close']).fillna(0)
        accuracy = accuracy_score(gt_labels, labels_wavelet)
        precision = precision_score(gt_labels, labels_wavelet, average='weighted')
        recall = recall_score(gt_labels, labels_wavelet, average='weighted')
        
        st.header("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        
        st.header("Charts")
        fig_wl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Denoised Data & Labels', 'Original Price'))
        fig_wl.add_trace(go.Scatter(x=df_wl.index, y=data_train_denoised, name='Wavelet Denoised Price', line=dict(color='orange')), row=1, col=1)
        up = df_wl[df_wl['label']==1]; down = df_wl[df_wl['label']==-1]
        fig_wl.add_trace(go.Scatter(x=up.index, y=up['close'], mode='markers', name='Up Label', marker=dict(color='green', symbol='triangle-up')), row=1, col=1)
        fig_wl.add_trace(go.Scatter(x=down.index, y=down['close'], mode='markers', name='Down Label', marker=dict(color='red', symbol='triangle-down')), row=1, col=1)
        fig_wl.add_trace(go.Scatter(x=df_wl.index, y=df_wl['close'], name='Original Price', line=dict(color='blue')), row=2, col=1)
        st.plotly_chart(fig_wl, use_container_width=True)
