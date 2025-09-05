import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import norm
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

st.title("QUANTITATIVE TRADING RESEARCH DASHBOARD")
st.markdown("An interactive dashboard for backtesting strategies and analyzing advanced market signals, now featuring **MLP-driven predictive signals** in the watchlist.")

# ==============================================================================
# Helper Functions (used across tabs)
# ==============================================================================
@st.cache_data
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.kraken()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if len(ohlcv) == 0: return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def ema(data, window):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

# ==============================================================================
# TAB STRUCTURE
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["🏆 Hawkes Strategy Backtester", "🔬 SG Swing Analysis", "🧠 MLP Predictive Watchlist"])

# ==============================================================================
# TAB 1: HAWKES STRATEGY BACKTESTER
# ==============================================================================
with tab1:
    # ... (Code for Tab 1 remains unchanged) ...
    st.header("Hawkes Process Momentum Strategy Backtest")
    st.sidebar.header("⚙️ Backtester Configuration")
    st.sidebar.subheader("Indicator & Data Settings")
    indicator_type = st.sidebar.selectbox("Indicator Type", ["HawkesBVC", "HawkesBSI"], key="bt_indicator")
    symbol_bt = st.sidebar.text_input("Symbol", "BTC/USD", key="bt_symbol")
    timeframe_bt = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1, key="bt_tf")
    data_limit_bt = st.sidebar.slider("Number of Data Bars", 500, 5000, 1500, key="bt_limit")
    st.sidebar.subheader("Hawkes Process Settings")
    kappa = st.sidebar.slider("Decay Factor (Kappa)", 0.01, 1.0, 0.1, 0.01, key="bt_kappa")
    if indicator_type == "HawkesBVC":
        volatility_window = st.sidebar.slider("BVC Volatility Window", 10, 100, 30, key="bt_vol_window")
    entry_threshold = st.sidebar.slider("Entry Threshold", 0.1, 5.0, 1.0, 0.1, key="bt_entry_thresh")
    st.sidebar.subheader("Backtest Settings")
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0, key="bt_cash")
    trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1, key="bt_size")

    @st.cache_data
    def calculate_hawkes_bsi(_df, kappa_decay):
        st.info("Simulating buy/sell volume and calculating Hawkes BSI..."); df_copy = _df.copy()
        price_change = df_copy['close'] - df_copy['open']; buy_ratio = np.where(price_change > 0, 0.75, 0.25)
        df_copy['buyvolume'] = df_copy['volume'] * buy_ratio; df_copy['sellvolume'] = df_copy['volume'] * (1 - buy_ratio)
        imbalance = df_copy['buyvolume'] - df_copy['sellvolume']; bsi = np.zeros(len(df_copy))
        for i in range(1, len(df_copy)): bsi[i] = bsi[i-1] * np.exp(-kappa_decay) + imbalance.iloc[i]
        df_copy['indicator'] = bsi
        return df_copy.iloc[1:]

    @st.cache_data
    def calculate_hawkes_bvc(_df, vol_window, kappa_decay):
        st.info("Classifying volume with BVC and calculating Hawkes BVC..."); df_copy = _df.copy()
        df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1)); df_copy['volatility'] = df_copy['return'].rolling(window=vol_window).std()
        df_copy['z_score'] = df_copy['return'] / (df_copy['volatility'] + 1e-9); df_copy['cdf'] = norm.cdf(df_copy['z_score'])
        df_copy['signed_volume'] = df_copy['volume'] * (2 * df_copy['cdf'] - 1)
        bvc = np.zeros(len(df_copy)); signed_volume_filled = df_copy['signed_volume'].fillna(0)
        for i in range(1, len(df_copy)): bvc[i] = bvc[i-1] * np.exp(-kappa_decay) + signed_volume_filled.iloc[i]
        df_copy['indicator'] = bvc
        return df_copy.iloc[vol_window:]

    @st.cache_data
    def run_hawkes_backtest(_df, cash, size, entry_thresh):
        st.info(f"Running backtest..."); position, equity, trades = 0.0, [cash], []
        for i in range(1, len(_df)):
            current_bar, prev_bar = _df.iloc[i], _df.iloc[i-1]; signal_current, signal_prev = current_bar['indicator'], prev_bar['indicator']
            if position > 0 and signal_prev > 0 and signal_current <= 0: cash += position * current_bar['open']; trades.append({'t': current_bar.name, 'type': 'EXIT LONG', 'p': current_bar['open']}); position = 0
            elif position < 0 and signal_prev < 0 and signal_current >= 0: cash += position * current_bar['open']; trades.append({'t': current_bar.name, 'type': 'EXIT SHORT', 'p': current_bar['open']}); position = 0
            if position == 0:
                if signal_prev < entry_thresh and signal_current >= entry_thresh: position += size; cash -= size * current_bar['close']; trades.append({'t': current_bar.name, 'type': 'BUY', 'p': current_bar['close']})
                elif signal_prev > -entry_thresh and signal_current <= -entry_thresh: position -= size; cash += size * current_bar['close']; trades.append({'t': current_bar.name, 'type': 'SELL', 'p': current_bar['close']})
            equity.append(cash + position * current_bar['close'])
        return pd.DataFrame(trades), pd.Series(equity[1:], index=_df.index[1:])

    if st.sidebar.button("🚀 Run Backtest", key="run_bt"):
        with st.spinner("Executing backtest pipeline... Please wait."):
            df_raw_bt = fetch_data(symbol_bt, timeframe_bt, data_limit_bt)
            if df_raw_bt.empty:
                st.error(f"Could not fetch data for {symbol_bt}. Please check the symbol and try again.")
            else:
                if indicator_type == "HawkesBSI": df_indicator = calculate_hawkes_bsi(df_raw_bt, kappa)
                else: df_indicator = calculate_hawkes_bvc(df_raw_bt, volatility_window, kappa)
                if df_indicator.empty: st.error("Error: The dataset is empty after feature calculation.")
                else:
                    trades_bt, equity_bt = run_hawkes_backtest(df_indicator, initial_cash, trade_size, entry_threshold)
                    st.success("✅ Backtest complete!")
                    st.subheader(f"Strategy: {indicator_type}"); final_equity = equity_bt.iloc[-1] if not equity_bt.empty else initial_cash; total_return = (final_equity / initial_cash - 1) * 100
                    st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades", len(trades_bt))

                    def plot_backtest_results(df_plot, equity_curve, trades, symbol_to_plot):
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Strategy Equity Curve', f'{indicator_type} Indicator', f'{symbol_to_plot} Price & Trades'), row_heights=[0.25, 0.25, 0.5])
                        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['indicator'], mode='lines', name=indicator_type, line=dict(color='orange')), row=2, col=1)
                        fig.add_hline(y=entry_threshold, line_width=2, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Buy Threshold"); fig.add_hline(y=-entry_threshold, line_width=2, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Sell Threshold")
                        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol_to_plot} Price', line=dict(color='blue')), row=3, col=1)
                        if not trades.empty:
                            buys = trades[trades['type'] == 'BUY']; sells = trades[trades['type'] == 'SELL']; exits = trades[trades['type'].str.contains('EXIT')]
                            fig.add_trace(go.Scatter(x=buys['t'], y=buys['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=3, col=1)
                            fig.add_trace(go.Scatter(x=sells['t'], y=sells['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=3, col=1)
                            fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Exits', marker=dict(color='black', symbol='x', size=8)), row=3, col=1)
                        fig.update_layout(height=900); return fig

                    st.plotly_chart(plot_backtest_results(df_indicator, equity_bt, trades_bt, symbol_bt), use_container_width=True)
                    with st.expander("Show Raw Trades Data"): st.dataframe(trades_bt)


# ==============================================================================
# TAB 2: SG SWING ANALYSIS
# ==============================================================================
with tab2:
    # ... (Code for Tab 2 remains unchanged) ...
    st.header("Savitzky-Golay Swing Point Detection with SpanB Overlay")
    st.markdown("This is an **offline analysis tool** for visualizing swing points. Note that this method has a **lookahead bias** and is **not** suitable for live trading signals.")
    st.sidebar.subheader("SG Swing Analysis Settings")
    symbol_sg = st.sidebar.text_input("Symbol", "BTC/USD", key="sg_symbol")
    window_short_sg = st.sidebar.slider("SG Fast Window", 5, 51, 33, step=2, key="sg_fast")
    window_long_sg = st.sidebar.slider("SG Slow Window", 101, 301, 257, step=2, key="sg_slow")
    polyorder_sg = st.sidebar.slider("SG Polyorder", 2, 5, 3, key="sg_poly")
    span_window_sg = st.sidebar.slider("SpanB Window", 20, 100, 52, key="sg_span")
    if st.sidebar.button("🔬 Run SG Analysis", key="run_sg"):
        df_sg = fetch_data(symbol_sg, '1h', 1000)
        if df_sg.empty or len(df_sg) < window_long_sg:
            st.error("Not enough data for the selected SG window length. Please select a shorter window or check the symbol.")
        else:
            close_prices = df_sg['close'].values
            smoothed_short = savgol_filter(close_prices, window_length=window_short_sg, polyorder=polyorder_sg)
            smoothed_long = savgol_filter(close_prices, window_length=window_long_sg, polyorder=polyorder_sg)
            diff_signal = smoothed_short - smoothed_long; threshold = np.std(diff_signal)
            swing_points = [i for i in range(1, len(diff_signal)-1) if np.abs(diff_signal[i])>threshold and np.abs(diff_signal[i])>np.abs(diff_signal[i-1]) and np.abs(diff_signal[i])>np.abs(diff_signal[i+1])]
            df_sg['spanB'] = (df_sg['high'].rolling(window=span_window_sg).max() + df_sg['low'].rolling(window=span_window_sg).min()) / 2
            fig_sg, ax_sg = plt.subplots(figsize=(15, 7)); segments = np.array([mdates.date2num(df_sg.index), close_prices]).T.reshape(-1, 1, 2); segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
            lc = LineCollection(segments, cmap='bwr', norm=plt.Normalize(vmin=-np.abs(diff_signal).max(), vmax=np.abs(diff_signal).max())); lc.set_array(diff_signal); ax_sg.add_collection(lc)
            ax_sg.plot(df_sg.index, smoothed_short, label=f"Savgol Short", color='gray', linestyle='--', lw=1)
            ax_sg.plot(df_sg.index, smoothed_long, label=f"Savgol Long", color='orange', linestyle='--', lw=1)
            ax_sg.scatter(df_sg.index[swing_points], close_prices[swing_points], color='lime', s=50, label="Swing Points", zorder=5)
            ax_sg.plot(df_sg.index, df_sg['spanB'], label="SpanB", color='purple', lw=2, linestyle='-.')
            ax_sg.set_xlim(df_sg.index.min(), df_sg.index.max()); ax_sg.set_ylim(close_prices.min()*0.98, close_prices.max()*1.02)
            ax_sg.legend(); ax_sg.set_title(f"SG Swing Point Analysis for {symbol_sg}"); plt.colorbar(lc, ax=ax_sg, label="SG Filter Difference")
            st.pyplot(fig_sg)


# ==============================================================================
# TAB 3: MLP PREDICTIVE WATCHLIST
# ==============================================================================
with tab3:
    st.header("MLP Predictive Watchlist")
    st.markdown("""
    This tool trains a unique Neural Network (MLP) for each cryptocurrency to generate predictive trading signals.
    - **Asset-Specific Models:** Each token gets its own model, trained on its unique historical data.
    - **Advanced Features:** Models consider momentum, volatility, trend strength (ADX), and volume to understand market context.
    - **`MLP Signal`**: The model's prediction: `Buy` (predicts price will hit profit target first), `Sell` (predicts price will hit stop-loss first), or `Hold`.
    - **`Confidence`**: The model's confidence in its prediction.
    """)
    st.sidebar.header("🧠 MLP Watchlist Configuration")
    min_volume_wl = st.sidebar.number_input("Minimum 24h Quote Volume", value=250000, key="wl_min_vol")
    
    STABLECOINS = {'USDC', 'DAI', 'BUSD', 'TUSD', 'USDT', 'UST'} # Simplified list

    @st.cache_data(ttl=3600) # Cache for 1 hour
    def get_filtered_tickers(min_quote_volume):
        st.info(f"Fetching tickers with > ${min_quote_volume:,} 24h volume...")
        try:
            exchange = ccxt.kraken(); tickers = exchange.fetch_tickers(); markets = exchange.load_markets()
            filtered_symbols = [
                symbol for symbol, ticker in tickers.items()
                if symbol.endswith('/USD') and
                ticker.get('quoteVolume') is not None and
                ticker['quoteVolume'] > min_quote_volume and
                markets.get(symbol, {}).get('base') not in STABLECOINS
            ]
            st.success(f"Found {len(filtered_symbols)} active tickers.")
            return filtered_symbols
        except Exception as e:
            st.error(f"Failed to fetch tickers: {e}")
            return []

    def get_adx(high, low, close, window):
        plus_dm = high.diff(); minus_dm = low.diff(-1)
        plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
        tr1 = pd.DataFrame(high - low); tr2 = pd.DataFrame(abs(high - close.shift(1))); tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
        minus_di = 100 * (abs(minus_dm.ewm(alpha=1/window, adjust=False).mean()) / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        return adx

    def get_triple_barrier_labels(close, lookahead_periods=5, tp_mult=1.5, sl_mult=1.5):
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std().fillna(method='bfill')
        
        labels = pd.Series(0, index=close.index) # Default to Hold
        
        for i in range(len(close) - lookahead_periods):
            entry_price = close.iloc[i]
            take_profit = entry_price * (1 + tp_mult * volatility.iloc[i])
            stop_loss = entry_price * (1 - sl_mult * volatility.iloc[i])
            
            future_prices = close.iloc[i+1 : i+1+lookahead_periods]
            
            # Check if take profit is hit first
            if (future_prices >= take_profit).any():
                labels.iloc[i] = 1 # Buy
            # Check if stop loss is hit first
            elif (future_prices <= stop_loss).any():
                labels.iloc[i] = -1 # Sell
        return labels

    @st.cache_data(ttl=3600 * 4) # Cache model predictions for 4 hours
    def get_mlp_prediction_for_asset(symbol, timeframe='1d', limit=1500):
        try:
            df = fetch_data(symbol, timeframe, limit)
            if df.empty or len(df) < 200: return "N/A", 0.0

            # --- 1. Feature Engineering ---
            df['ret_fast'] = df['close'].pct_change(7)
            df['ret_slow'] = df['close'].pct_change(30)
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['adx'] = get_adx(df['high'], df['low'], df['close'], 14)
            df['volume_z'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
            
            # For beta, we need BTC data
            btc_df = fetch_data('BTC/USD', timeframe, limit)
            if not btc_df.empty:
                market_ret = btc_df['close'].pct_change()
                asset_ret = df['close'].pct_change()
                rolling_cov = asset_ret.rolling(window=30).cov(market_ret)
                rolling_var = market_ret.rolling(window=30).var()
                df['beta'] = rolling_cov / rolling_var
            else:
                df['beta'] = 1.0 # Default if BTC data fails

            feature_names = ['ret_fast', 'ret_slow', 'volatility', 'adx', 'volume_z', 'beta']
            df.dropna(inplace=True)
            X = df[feature_names]
            
            if len(X) < 100: return "Data Insufficient", 0.0

            # --- 2. Labeling ---
            y = get_triple_barrier_labels(df['close'])
            y = y.loc[X.index] # Align labels with features

            # --- 3. Model Training ---
            # Use last 80% of data for training, ensuring we don't use future data
            train_size = int(len(X) * 0.8)
            X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
            
            if len(X_train) < 50: return "Train Data Insufficient", 0.0

            pipeline = make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42, early_stopping=True)
            )
            pipeline.fit(X_train, y_train)

            # --- 4. Prediction ---
            latest_features = X.iloc[-1:]
            prediction_code = pipeline.predict(latest_features)[0]
            probabilities = pipeline.predict_proba(latest_features)[0]
            
            signal_map = {1: "Buy", -1: "Sell", 0: "Hold"}
            signal = signal_map.get(prediction_code, "Hold")
            confidence = probabilities.max()
            
            return signal, confidence

        except Exception as e:
            # st.warning(f"Could not process {symbol}: {e}") # Too noisy for UI
            return "Error", 0.0

    if st.sidebar.button("🧠 Run MLP Watchlist Analysis", key="run_wl"):
        watchlist_symbols = get_filtered_tickers(min_volume_wl)
        
        if not watchlist_symbols:
            st.error("No tickers met the filter criteria. Watchlist is empty.")
        else:
            results = []
            progress_bar = st.progress(0, text="Initializing MLP model training...")
            
            for i, symbol in enumerate(watchlist_symbols):
                signal, confidence = get_mlp_prediction_for_asset(symbol)
                results.append({
                    'Token': symbol,
                    'MLP Signal': signal,
                    'Confidence': confidence
                })
                progress_bar.progress((i + 1) / len(watchlist_symbols), text=f"Training & Predicting for {symbol}...")
            
            progress_bar.empty()
            df_watchlist = pd.DataFrame(results)

            # --- Display Results ---
            st.subheader("🤖 AI-Driven Market Signals")
            
            # Sort by confidence for more actionable insights
            df_watchlist = df_watchlist.sort_values(by='Confidence', ascending=False).reset_index(drop=True)
            
            df_watchlist['Confidence'] = df_watchlist['Confidence'].map('{:.1%}'.format)

            st.dataframe(df_watchlist, use_container_width=True, hide_index=True)
