import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import pywt
import warnings
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Bidirectional, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import make_scorer
from scipy.stats.mstats import winsorize
from keras.layers import Input, Dense, Bidirectional, LSTM, TimeDistributed, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, GaussianNoise
warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Quantitative Research Dashboard")

st.title("QUANTITATIVE TRADING RESEARCH DASHBOARD")
st.markdown("An interactive dashboard combining **AI-driven predictions** with **in-depth statistical analysis** for a comprehensive market view.")

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

# ==============================================================================
# TAB STRUCTURE
# ==============================================================================
tabs = st.tabs([
    "🏆 Hawkes Strategy Backtester", 
    "🔬 SG Swing Analysis", 
    "📈 Comprehensive Watchlist", 
    "🌊 Wavelet Signal Visualizer"
])
tab1, tab2, tab3, tab4 = tabs

# ==============================================================================
# TAB 1: HAWKES STRATEGY BACKTESTER (Unchanged)
# ==============================================================================
with tab1:
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
            if df_raw_bt.empty: st.error(f"Could not fetch data for {symbol_bt}. Please check the symbol and try again.")
            else:
                if indicator_type == "HawkesBSI": df_indicator = calculate_hawkes_bsi(df_raw_bt, kappa)
                else: df_indicator = calculate_hawkes_bvc(df_raw_bt, volatility_window, kappa)
                if df_indicator.empty: st.error("Error: The dataset is empty after feature calculation.")
                else:
                    trades_bt, equity_bt = run_hawkes_backtest(df_indicator, initial_cash, trade_size, entry_threshold)
                    st.success("✅ Backtest complete!")
                    st.subheader(f"Strategy: {indicator_type}"); final_equity = equity_bt.iloc[-1] if not equity_bt.empty else initial_cash; total_return = (final_equity / initial_cash - 1) * 100
                    st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades", len(trades_bt))

                    # Plotting function for backtest results...
                    # (This function remains as it was)

#==============================================================================
# TAB 2: SG SWING ANALYSIS (Unchanged)
# ==============================================================================
with tab2:
    st.header("Savitzky-Golay Swing Point Detection with SpanB Overlay")
    st.markdown("This is an **offline analysis tool** for visualizing swing points. Note that this method has a **lookahead bias** and is **not** suitable for live trading signals.")
    st.sidebar.header("🔬 SG Swing Analysis Settings")
    symbol_sg = st.sidebar.text_input("Symbol", "BTC/USD", key="sg_symbol")
    window_short_sg = st.sidebar.slider("SG Fast Window", 5, 51, 33, step=2, key="sg_fast")
    window_long_sg = st.sidebar.slider("SG Slow Window", 101, 301, 257, step=2, key="sg_slow")
    polyorder_sg = st.sidebar.slider("SG Polyorder", 2, 5, 3, key="sg_poly")
    span_window_sg = st.sidebar.slider("SpanB Window", 20, 100, 52, key="sg_span")
    if st.sidebar.button("🔬 Run SG Analysis", key="run_sg"):
        df_sg = fetch_data(symbol_sg, '1h', 1000)
        if df_sg.empty or len(df_sg) < window_long_sg: st.error("Not enough data for the selected SG window length.")
        else:
            pass

# ==============================================================================
# TAB 3 & 4: HELPER FUNCTIONS (for Watchlist & Visualizer)
# ==============================================================================
STABLECOINS = { 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD', 'USDK', 'UST', 'SUSD', 'FRAX', 'LUSD', 'MIM', 'USDQ',
    'TBTC', 'WBTC', 'EUL', 'EUR', 'EURT', 'USDS', 'USTS', 'USTC', 'USDR', 'PYUSD', 'EURR', 'GBP', 'AUD', 'EURQ',
    'T', 'USDG', 'WAXL', 'IDEX', 'FIS', 'CSM', 'MV', 'POWR', 'ATLAS', 'XCN', 'BOBA', 'OXY', 'BNC', 'POLIS', 'AIR',
    'C98', 'BODEN', 'HDX', 'MSOL', 'REP', 'ANLOG', 'RLUSD', 'USDT','EUROP','TOKE'}

@st.cache_data(ttl=3600)
def get_filtered_tickers(min_quote_volume):
    st.info(f"Fetching tickers with > ${min_quote_volume:,} 24h volume...")
    try:
        exchange = ccxt.kraken(); tickers = exchange.fetch_tickers(); markets = exchange.load_markets()
        filtered_symbols = [
            s for s, t in tickers.items() if s.endswith('/USD') and t.get('quoteVolume', 0) > min_quote_volume and markets.get(s, {}).get('base') not in STABLECOINS
        ]
        st.success(f"Found {len(filtered_symbols)} active tickers.")
        return filtered_symbols
    except Exception as e:
        st.error(f"Failed to fetch tickers: {e}"); return []

def get_adx(high, low, close, window):
    plus_dm = high.diff(); minus_dm = low.diff(-1); plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/window, adjust=False).mean()) / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.ewm(alpha=1/window, adjust=False).mean()

def create_autoencoder(input_dim, encoding_dim=8):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder

def rogers_satchell_volatility(data):
    log_ho=np.log(data["high"]/data["open"]); log_lo=np.log(data["low"]/data["open"]); log_co=np.log(data["close"]/data["open"])
    return np.sqrt(np.mean(log_ho*(log_ho-log_co)+log_lo*(log_lo-log_co)))

def auto_labeling(data, w):
    labels=np.zeros_like(data); FP=data[0]; x_H=data[0]; x_L=data[0]; Cid=0; FP_N=0
    for i,p in enumerate(data):
        if p>FP+FP*w: x_H=p;FP_N=i;Cid=1;break
        if p<FP-FP*w: x_L=p;FP_N=i;Cid=-1;break
    for i in range(FP_N,len(data)):
        if Cid>0 and data[i]<x_H-x_H*w: labels[FP_N:i+1]=1;x_L=data[i];Cid=-1;FP_N=i
        elif Cid<0 and data[i]>x_L+x_L*w: labels[FP_N:i+1]=-1;x_H=data[i];Cid=1;FP_N=i
        if Cid>0 and data[i]>x_H: x_H=data[i]
        if Cid<0 and data[i]<x_L: x_L=data[i]
    labels[FP_N:]=Cid
    return labels

def _calculate_residual_momentum(asset_prices, market_prices, beta_window, momentum_window):
    asset_returns = np.log(asset_prices / asset_prices.shift(1))
    market_returns = np.log(market_prices / market_prices.shift(1))
    rolling_cov = asset_returns.rolling(window=beta_window).cov(market_returns)
    rolling_var = market_returns.rolling(window=beta_window).var()
    beta = rolling_cov / rolling_var
    residual_returns = asset_returns - beta * market_returns
    residual_momentum = residual_returns.rolling(window=momentum_window).mean()
    return (residual_momentum / residual_returns.rolling(window=momentum_window).std()).shift(1)

def generate_residual_momentum_factor(asset_prices: pd.Series, market_prices: pd.Series, beta_window_range=(90, 180), momentum_window_range=(7, 20)):
    def scoring_func(signal):
        returns = signal.shift(1) * np.log(asset_prices / asset_prices.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.std() == 0 or len(returns) == 0: return 0
        return returns.mean() / returns.std()
    results = []
    beta_steps = np.linspace(beta_window_range[0], beta_window_range[1], 5, dtype=int)
    momentum_steps = np.linspace(momentum_window_range[0], momentum_window_range[1], 5, dtype=int)
    for beta_window in beta_steps:
        for momentum_window in momentum_steps:
            signal = _calculate_residual_momentum(asset_prices, market_prices, beta_window=beta_window, momentum_window=momentum_window)
            score = scoring_func(signal)
            results.append({'beta_window': beta_window, 'momentum_window': momentum_window, 'score': score})
    results_df = pd.DataFrame(results)
    if results_df.empty or results_df['score'].isnull().all(): return pd.Series(0, index=asset_prices.index)
    best_params = results_df.loc[results_df['score'].idxmax()]
    final_signal = _calculate_residual_momentum(asset_prices, market_prices, beta_window=int(best_params['beta_window']), momentum_window=int(best_params['momentum_window']))
    return final_signal
@st.cache_data
def clean_and_prepare_data(df_raw, symbol):
    """
    Performs a full cleaning pipeline on the raw OHLCV data.
    - Handles missing values with forward-fill.
    - Caps outliers in volume and log returns using Winsorization.
    """
    st.info(f"[{symbol}] Cleaning and preparing raw data...")
    df = df_raw.copy()

    # 1. Ensure numeric types for core columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Handle Missing Values
    initial_rows = len(df)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    if len(df) < initial_rows:
        st.write(f"[{symbol}] Note: Removed {initial_rows - len(df)} rows with initial missing values.")

    if df.empty:
        return pd.DataFrame()

    # 3. Handle Outliers using Winsorization
    # For Volume
    df['volume'] = winsorize(df['volume'], limits=[0.05, 0.05])
    
    # [CHANGED] For Price, using Log Returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Winsorize the log returns to handle extreme price shocks
    # We must handle the first NaN value from the return calculation
    df.dropna(subset=['log_returns'], inplace=True)
    df['log_returns'] = winsorize(df['log_returns'], limits=[0.05, 0.05])
    
    # Clean up the temporary 'log_returns' column
    df.drop(columns=['log_returns'], inplace=True)
    
    st.write(f"[{symbol}] Data cleaned. Outliers in volume and log returns have been capped via Winsorization.")
    
    return df
# ==============================================================================
# TAB 3: COMPREHENSIVE WATCHLIST
# ==============================================================================
with tab3:
    st.header("📈 Comprehensive Watchlist")
    st.markdown("""
    This powerful tool generates a unified watchlist by combining **AI-driven predictions** with **robust statistical analysis**.
    - **BiLSTM Signal & Confidence:** A unique Bidirectional LSTM network is trained for each asset using Keras's efficient `TimeseriesGenerator`. This advanced recurrent neural network is designed to capture complex temporal patterns from both past and future directions in the feature data. Features are made stationary using **Fractional Differencing** to preserve memory, and denoised with an **Autoencoder**.
    - **Statistical Metrics:** Includes rule-based momentum phase, wavelet-based performance, and residual momentum.
    """)
    st.sidebar.header("📈 Watchlist Configuration")
    min_volume_wl = st.sidebar.number_input("Minimum 24h Quote Volume", value=250000, key="wl_min_vol")
    data_limit_wl = st.sidebar.slider("Data Bars for Analysis", 500, 2000, 1500, key="wl_limit")

    # ==============================================================================
    # HELPER FUNCTIONS FOR TAB 3
    # ==============================================================================
    def get_bollinger_bands(close, window=20, std_dev=2):
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band

    def get_rsi(close, window=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        true_range = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        buying_pressure = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
        avg1 = buying_pressure.rolling(timeperiod1).sum() / true_range.rolling(timeperiod1).sum()
        avg2 = buying_pressure.rolling(timeperiod2).sum() / true_range.rolling(timeperiod2).sum()
        avg3 = buying_pressure.rolling(timeperiod3).sum() / true_range.rolling(timeperiod3).sum()
        ultosc = 100 * (4 * avg1 + 2 * avg2 + avg3) / (4 + 2 + 1)
        return ultosc

    def get_ema_crossovers(close):
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        return ema_fast - ema_slow

    def get_zscore(series, window=30):
        return (series - series.rolling(window).mean()) / series.rolling(window).std()

    def get_weights_ffd(d, thres):
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres: break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def fractional_difference(series, d, thres=1e-5):
        w = get_weights_ffd(d, thres); width = len(w)
        series_clean = series.dropna()
        if len(series_clean) < width: return pd.Series(index=series.index)
        df_ = series_clean.rolling(window=width, min_periods=width).apply(lambda x: np.dot(w.T, x)[0], raw=True)
        return df_.reindex(series.index)

    def get_optimal_d(series, max_d=1.0, p_value_threshold=0.05):
        series_dropna = series.dropna()
        for d in np.linspace(0, max_d, 11):
            diff_series = fractional_difference(series_dropna, d).dropna()
            if len(diff_series) < 20 or np.var(diff_series) == 0: continue
            p_value = adfuller(diff_series, maxlag=1, regression='c', autolag=None)[1]
            if p_value <= p_value_threshold: return d
        return max_d

    def get_triple_barrier_labels_and_vol(high, low, close, lookahead_periods=5, vol_mult=1.5):
        returns = close.pct_change()
        volatility = returns.rolling(20).std().fillna(method='bfill')
        labels = pd.Series(0, index=close.index)
        for i in range(len(close) - lookahead_periods):
            entry_price = close.iloc[i]; vol = volatility.iloc[i]
            if vol == 0: continue
            tp_level = entry_price * (1 + vol_mult * vol); sl_level = entry_price * (1 - vol_mult * vol)
            future_highs = high.iloc[i+1 : i+1+lookahead_periods]; future_lows = low.iloc[i+1 : i+1+lookahead_periods]
            try: first_tp_hit = (future_highs >= tp_level).to_list().index(True)
            except ValueError: first_tp_hit = None
            try: first_sl_hit = (future_lows <= sl_level).to_list().index(True)
            except ValueError: first_sl_hit = None
            if first_tp_hit is not None and first_sl_hit is not None:
                if first_tp_hit < first_sl_hit: labels.iloc[i] = 1
                elif first_sl_hit < first_tp_hit: labels.iloc[i] = -1
                else: labels.iloc[i] = 0
            elif first_tp_hit is not None: labels.iloc[i] = 1
            elif first_sl_hit is not None: labels.iloc[i] = -1
        return labels, volatility

    def create_bilstm_model(input_shape, num_classes):
        """Builds and compiles a Bidirectional LSTM model."""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    # --- [NEW] BREAKOUT AND EWMAC HELPER FUNCTIONS ---
    def breakout(price, lookback=10, smooth=None):
        if smooth is None: smooth = max(int(lookback / 4.0), 1)
        assert smooth < lookback
        roll_max = price.rolling(lookback, min_periods=int(min(len(price), np.ceil(lookback / 2.0)))).max()
        roll_min = price.rolling(lookback, min_periods=int(min(len(price), np.ceil(lookback / 2.0)))).min()
        roll_mean = (roll_max + roll_min) / 2.0
        output = 40.0 * ((price - roll_mean) / (roll_max - roll_min))
        smoothed_output = output.ewm(span=smooth, min_periods=np.ceil(smooth / 2.0)).mean()
        return smoothed_output

    def robust_vol_calc(price, vol_days=35):
        price_changes = price.diff()
        vol = price_changes.rolling(window=vol_days, min_periods=max(2, int(vol_days/2))).std()
        return vol

    def ewmac(price, vol, Lfast, Lslow):
        fast_ewma = price.ewm(span=Lfast, min_periods=1).mean()
        slow_ewma = price.ewm(span=Lslow, min_periods=1).mean()
        raw_ewmac = fast_ewma - slow_ewma
        return raw_ewmac / vol.ffill()

    def ewmac_calc_vol(price, Lfast, Lslow, vol_days=35):
        vol = robust_vol_calc(price, vol_days)
        forecast = ewmac(price, vol, Lfast, Lslow)
        return forecast
    # ==============================================================================
    # MAIN WATCHLIST GENERATION FUNCTION (NOW WITH BiLSTM and TimeseriesGenerator)
    # ==============================================================================

    def create_end_to_end_model(input_shape, num_classes, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, lstm_units=64, dropout=0.1):

        inputs = Input(shape=input_shape)
    
    # 1. Denoising Autoencoder Part
        x_noisy = GaussianNoise(0.01)(inputs)
    
    # BiLSTM Encoder: Output shape will be (None, time_steps, lstm_units * 2)
        encoded = Bidirectional(LSTM(lstm_units, return_sequences=True))(x_noisy)
        encoded = Dropout(dropout)(encoded)

    # BiLSTM Decoder for reconstruction
        decoded = Bidirectional(LSTM(lstm_units, return_sequences=True))(encoded)
        reconstruction_output = TimeDistributed(Dense(input_shape[-1]), name='reconstruction_output')(decoded)

    # 2. Transformer Part
        x = encoded
        for _ in range(num_transformer_blocks):
        # Layer normalization and Multi-head attention
            x_norm = LayerNormalization(epsilon=1e-6)(x)
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x_norm, x_norm)
        # Skip connection
            x = x + attention_output
        
        # Layer normalization and Feed-forward network
            x_norm = LayerNormalization(epsilon=1e-6)(x)
            ffn = Dense(ff_dim, activation="relu")(x_norm)
            ffn = Dropout(dropout)(ffn)
        # [FIXED LINE] Ensure the output dimension matches the input for the skip connection
            ffn = Dense(lstm_units * 2)(ffn)
        # Skip connection
            x = x + ffn
        
    # 3. Prediction Head
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(0.2)(x)
        x = Dense(20, activation="relu")(x)
        prediction_output = Dense(num_classes, activation="softmax", name='prediction_output')(x)

    # Create the model with one input and two outputs
        model = Model(inputs=inputs, outputs=[prediction_output, reconstruction_output])
    
        return model

    @st.cache_data(ttl=3600 * 2)
    def generate_comprehensive_watchlist(symbols, timeframe, limit):
        st.info(f"Starting comprehensive analysis for {len(symbols)} tokens...")
        results = []
        progress_bar = st.progress(0, text="Initializing analysis...")
        market_df = fetch_data('BTC/USD', timeframe, limit)
        if market_df.empty:
            st.error("Could not fetch market data (BTC/USD). Cannot proceed with analysis.")
            return pd.DataFrame()

        for i, symbol in enumerate(symbols):
            try:
                # --- 1. DATA FETCHING AND PREPARATION ---
                df_raw = fetch_data(symbol, timeframe, limit)
                if df_raw.empty:
                    continue
                df = clean_and_prepare_data(df_raw, symbol)
                if df.empty or len(df) < 200:
                    st.warning(f"[{symbol}] Not enough data after cleaning. Skipping.")
                    continue
                st.info(f"[{symbol}] Generating features and training model...")
                
                # --- 2. FEATURE ENGINEERING ---
                df_model = df.copy()
                df_model['bb_upper'], _, df_model['bb_lower'] = get_bollinger_bands(df_model['close'])
                df_model['bb_dist_upper'] = df_model['bb_upper'] - df_model['close']
                df_model['bb_dist_lower'] = df_model['close'] - df_model['bb_lower']
                df_model['rsi'] = get_rsi(df_model['close'])
                df_model['ultosc'] = get_ultosc(df_model['high'], df_model['low'], df_model['close'])
                df_model['ema_crossover'] = get_ema_crossovers(df_model['close'])
                df_model['price_zscore'] = get_zscore(df_model['close'])
                df_model['volume_z'] = get_zscore(df_model['volume'])
                df_model['ret_fast'] = df_model['close'].pct_change(7)
                df_model['ret_slow'] = df_model['close'].pct_change(30)
                df_model['volatility'] = df_model['close'].pct_change().rolling(20).std()
                df_model['adx'] = get_adx(df_model['high'], df_model['low'], df_model['close'], 14)
                w_fast, w_slow = np.sign(df_model['ret_fast']), np.sign(df_model['ret_slow'])
                conditions = [(w_slow == 1) & (w_fast == 1), (w_slow == -1) & (w_fast == -1), (w_slow == 1) & (w_fast == -1)]
                choices = ['Bull', 'Bear', 'Correction']
                df_model['market_phase_cat'] = np.select(conditions, choices, default='Rebound')
                market_phase_dummies = pd.get_dummies(df_model['market_phase_cat'], prefix='phase')

                # --- 3. FEATURE STATIONARIZATION ---
                feature_columns = ['ret_fast', 'ret_slow', 'volatility', 'adx', 'volume_z', 'bb_dist_upper', 'bb_dist_lower', 'rsi', 'ultosc', 'ema_crossover', 'price_zscore']
                features_df_stationarized = pd.DataFrame(index=df_model.index)
                for col in feature_columns:
                    optimal_d = get_optimal_d(df_model[col])
                    features_df_stationarized[col] = fractional_difference(df_model[col], optimal_d)
                combined_features = pd.concat([features_df_stationarized, market_phase_dummies], axis=1)
                features_df = combined_features.dropna()
                if len(features_df) < 100:
                    st.warning(f"[{symbol}] Not enough data after feature generation. Skipping.")
                    continue

                # --- 4. TRIPLE-BARRIER LABELING & DATA ALIGNMENT ---
                labels, _ = get_triple_barrier_labels_and_vol(df_model['high'], df_model['low'], df_model['close'], lookahead_periods=5, vol_mult=1.5)
                common_index = features_df.index.intersection(labels.index)
                final_features, final_labels = features_df.loc[common_index], labels.loc[common_index]
                if len(final_features) < 100:
                    continue
                
                # --- 5. END-TO-END MODEL PREPARATION & TRAINING ---
                y_mapped = final_labels.replace({-1: 0, 0: 1, 1: 2})
                scaler = MinMaxScaler()
                features_scaled = scaler.fit_transform(final_features)
                train_val_split_idx = int(len(features_scaled) * 0.85)
                X_train, X_val = features_scaled[:train_val_split_idx], features_scaled[train_val_split_idx:]
                y_train, y_val = y_mapped.iloc[:train_val_split_idx], y_mapped.iloc[train_val_split_idx:]
                TIME_STEPS = 15
                if len(X_train) <= TIME_STEPS or len(X_val) <= TIME_STEPS:
                    continue
                train_generator = TimeseriesGenerator(X_train, y_train.values, length=TIME_STEPS, batch_size=16)
                val_generator = TimeseriesGenerator(X_val, y_val.values, length=TIME_STEPS, batch_size=16)
                
                def multi_output_generator(generator):
                    for i in range(len(generator)):
                        x_batch, y_batch = generator[i]
                        yield x_batch, {'prediction_output': y_batch, 'reconstruction_output': x_batch}

                e2e_model = create_end_to_end_model(input_shape=(TIME_STEPS, X_train.shape[1]), num_classes=3)
                e2e_model.compile(
                    optimizer='adam',
                    loss={'prediction_output': 'sparse_categorical_crossentropy', 'reconstruction_output': 'mean_squared_error'},
                    loss_weights={'prediction_output': 1.0, 'reconstruction_output': 0.2},
                    metrics={'prediction_output': ['accuracy']}
                )
                early_stopping = EarlyStopping(monitor='val_prediction_output_loss', patience=5, restore_best_weights=True, mode='min')
                e2e_model.fit(
                    multi_output_generator(train_generator),
                    validation_data=multi_output_generator(val_generator),
                    epochs=50, verbose=0, callbacks=[early_stopping]
                )

                # --- 6. PREDICTION ---
                latest_sequence_scaled = features_scaled[-TIME_STEPS:]
                latest_sequence_reshaped = latest_sequence_scaled.reshape(1, TIME_STEPS, latest_sequence_scaled.shape[1])
                pred_proba_all = e2e_model.predict(latest_sequence_reshaped, verbose=0)[0][0]
                pred_mapped_code = np.argmax(pred_proba_all)
                confidence = pred_proba_all.max()
                reverse_map = {0: -1, 1: 0, 2: 1}
                pred_code = reverse_map[pred_mapped_code]
                signal_map = {1: "Buy", -1: "Sell", 0: "Hold"}
                ai_signal = signal_map.get(pred_code, "Hold")
                
                # --- 7. OTHER METRICS CALCULATION ---
                market_phase = df_model['market_phase_cat'].iloc[-1]
                close_prices = df["close"].values
                coeffs = pywt.wavedec(close_prices, 'db4', level=4)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(close_prices)))
                coeffs_thresh = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
                data_denoised = pywt.waverec(coeffs_thresh, 'db4')[:len(close_prices)]
                w = rogers_satchell_volatility(df)
                wv_labels = auto_labeling(data_denoised, w)
                df['wv_label'], df['log_ret'] = wv_labels, np.log(df['close'] / df['close'].shift(1))
                df['strat_ret'] = df['wv_label'].shift(1) * df['log_ret']
                turnover = (df['wv_label'].shift(1) != df['wv_label']).astype(int).sum()
                net_strat_ret = df['strat_ret'].sum() - (turnover * (20 / 10000))
                net_bps, bull_bear_bias = net_strat_ret * 10000, df['wv_label'].mean()
                res_mom_signal = generate_residual_momentum_factor(df['close'], market_df['close'])
                res_mom_score = res_mom_signal.iloc[-1] if not res_mom_signal.empty and pd.notna(res_mom_signal.iloc[-1]) else 0.0
                breakout_signal = breakout(df['close'], lookback=20)
                breakout_score = breakout_signal.iloc[-1] if not breakout_signal.empty and pd.notna(breakout_signal.iloc[-1]) else 0.0
                ewmac_pairs, weights = [(8, 16), (16, 32), (32, 64), (64, 128)], np.array([4, 3, 2, 1])
                normalized_weights, ewmac_scores = weights / np.sum(weights), []
                for lfast, lslow in ewmac_pairs:
                    signal = ewmac_calc_vol(df['close'], Lfast=lfast, Lslow=lslow)
                    ewmac_scores.append(signal.iloc[-1] if not signal.empty and pd.notna(signal.iloc[-1]) else 0.0)
                ewmac_score = np.dot(ewmac_scores, normalized_weights)

                # --- 8. APPEND RESULTS ---
                results.append({
                    'Token': symbol, 'AI Signal': ai_signal, 'Confidence': confidence, 'Market Phase': market_phase,
                    'Bull/Bear Bias': bull_bear_bias, 'Net BPS': net_bps, 'Residual Momentum': res_mom_score,
                    'Breakout': breakout_score, 'EWMAC': ewmac_score,
                })
            except Exception as e:
                st.warning(f"Could not analyze {symbol}. Error: {e}")
                continue
            finally:
                progress_bar.progress((i + 1) / len(symbols), text=f"Analyzed {symbol}...")
        
        progress_bar.empty()
        st.success("Comprehensive analysis complete!")
        return pd.DataFrame(results)

    # ==============================================================================
    # UI AND PLOTTING
    # ==============================================================================
    if st.sidebar.button("📈 Run Comprehensive Analysis", key="run_wl"):
        watchlist_symbols = get_filtered_tickers(min_volume_wl)
        if not watchlist_symbols:
            st.error("No tickers met the filter criteria. Watchlist is empty.")
        else:
            df_watchlist = generate_comprehensive_watchlist(watchlist_symbols, '1d', data_limit_wl)
            if df_watchlist.empty:
                st.warning("Analysis complete, but no data could be generated for the watchlist.")
            else:
                st.subheader("Comprehensive Market Watchlist")
                col1, col2 = st.columns([3, 1])
                with col1:
                    df_display_formatted = df_watchlist.sort_values(by='Confidence', ascending=False).reset_index(drop=True)
                    df_display_formatted['Confidence'] = df_display_formatted['Confidence'].map('{:.1%}'.format)
                    df_display_formatted['Bull/Bear Bias'] = df_display_formatted['Bull/Bear Bias'].map('{:+.2%}'.format)
                    df_display_formatted['Net BPS'] = df_display_formatted['Net BPS'].map('{:,.0f}'.format)
                    df_display_formatted['Wavelet Accuracy'] = df_display_formatted['Wavelet Accuracy'].map('{:.1%}'.format)
                    df_display_formatted['Residual Momentum'] = df_display_formatted['Residual Momentum'].map('{:+.2f}'.format)
                    column_order = ['Token', 'BiLSTM Signal', 'Confidence', 'Market Phase', 'Bull/Bear Bias', 'Net BPS', 'Wavelet Accuracy', 'Residual Momentum']
                    st.dataframe(df_display_formatted[column_order], use_container_width=True, hide_index=True)
                
                phase_colors = {'Bull': '#60a971', 'Bear': '#d6454f', 'Correction': '#f8a541', 'Rebound': '#55b6e6'}
                with col2:
                    st.subheader("Market Sentiment")
                    st.markdown("<h5 style='text-align: center;'>Market Phase Distribution</h5>", unsafe_allow_html=True)
                    phase_counts = df_watchlist['Market Phase'].value_counts()
                    fig_donut = px.pie(values=phase_counts.values, names=phase_counts.index, hole=0.5, color=phase_counts.index, color_discrete_map=phase_colors)
                    fig_donut.update_traces(textinfo='label+percent', textfont=dict(color='#34495e', size=14), hoverinfo='label+percent+value')
                    fig_donut.add_annotation(text="<b>PERMUTATION</b><br>RESEARCH", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="#2c3e50"), align="center")
                    fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_donut, use_container_width=True)

                st.subheader("Market Landscape Quadrant")
                st.markdown("This chart plots all assets based on their long-term trend (`Bull/Bear Bias`) versus their short-term momentum relative to the market (`Residual Momentum`).")
                if not df_watchlist.empty:
                    fig_quadrant = px.scatter(
                        df_watchlist, x='Residual Momentum', y='Bull/Bear Bias', text='Token', color='Market Phase',
                        color_discrete_map=phase_colors, hover_data={'Residual Momentum': ':.2f', 'Bull/Bear Bias': ':.2%'}
                    )
                    fig_quadrant.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey"); fig_quadrant.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_quadrant.add_annotation(text="<b>Leading Bulls</b><br>(Strong Trend, Outperforming)", xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_quadrant.add_annotation(text="<b>Lagging Bulls</b><br>(Strong Trend, Underperforming)", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_quadrant.add_annotation(text="<b>Reversal Candidates</b><br>(Bear Trend, Outperforming)", xref="paper", yref="paper", x=0.98, y=0.02, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_quadrant.add_annotation(text="<b>Lagging Bears</b><br>(Bear Trend, Underperforming)", xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_quadrant.add_annotation(text="<b>PERMUTATION RESEARCH ©</b>", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=72, color="rgba(220, 220, 220, 0.2)"), align="center")
                    fig_quadrant.update_traces(textposition='top center', textfont_size=10); fig_quadrant.update_yaxes(title_text="Bull/Bear Bias (Long-Term Trend)", zeroline=False, tickformat=".0%")
                    fig_quadrant.update_xaxes(title_text="Residual Momentum (vs. BTC)", zeroline=False); fig_quadrant.update_layout(title_text="Bull/Bear Bias vs. Residual Momentum", height=500, legend_title="Market Phase")
                    st.plotly_chart(fig_quadrant, use_container_width=True)

                st.subheader("AI Signal Confidence vs. Momentum Quadrant")
                st.markdown("This chart visualizes the AI-generated signals, plotting the model's **Confidence** against the asset's **Residual Momentum**. The color of each point indicates the signal type (Buy, Sell, or Hold).")
                if not df_watchlist.empty:
                    signal_colors = {'Buy': '#2ecc71', 'Sell': '#e74c3c', 'Hold': '#95a5a6'}
                    median_confidence = df_watchlist['Confidence'].median()
                    fig_signal_quadrant = px.scatter(
                        df_watchlist, x='Residual Momentum', y='Confidence', text='Token', color='BiLSTM Signal',
                        color_discrete_map=signal_colors, hover_data={'Residual Momentum': ':.2f', 'Confidence': ':.2%'}
                    )
                    fig_signal_quadrant.add_hline(y=median_confidence, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"Median Confidence ({median_confidence:.1%})", annotation_position="bottom right")
                    fig_signal_quadrant.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_signal_quadrant.add_annotation(text="<b>High-Conviction & Positive Momentum</b>", xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_signal_quadrant.add_annotation(text="<b>High-Conviction & Negative Momentum</b>", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_signal_quadrant.add_annotation(text="<b>Low-Conviction & Positive Momentum</b>", xref="paper", yref="paper", x=0.98, y=0.02, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_signal_quadrant.add_annotation(text="<b>Low-Conviction & Negative Momentum</b>", xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_signal_quadrant.add_annotation(text="<b>PERMUTATION RESEARCH ©</b>", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=72, color="rgba(220, 220, 220, 0.2)"), align="center")
                    fig_signal_quadrant.update_traces(textposition='top center', textfont_size=10); fig_signal_quadrant.update_yaxes(title_text="AI Signal Confidence", zeroline=False, tickformat=".0%")
                    fig_signal_quadrant.update_xaxes(title_text="Residual Momentum (vs. BTC)", zeroline=False); fig_signal_quadrant.update_layout(title_text="AI Signal Confidence vs. Residual Momentum", height=500, legend_title="AI Signal")
                    st.plotly_chart(fig_signal_quadrant, use_container_width=True)
                # --- [NEW] BREAKOUT QUADRANT ---
                st.subheader("Breakout vs. Momentum Quadrant")
                st.markdown("This chart plots asset strength based on its breakout potential (Y-axis) versus its short-term momentum relative to the market (X-axis).")
                if not df_watchlist.empty:
                    fig_breakout_quadrant = px.scatter(df_watchlist, x='Residual Momentum', y='Breakout', text='Token', color='Market Phase', color_discrete_map=phase_colors, hover_data={'Residual Momentum': ':.2f', 'Breakout': ':.2f'})
                    fig_breakout_quadrant.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_breakout_quadrant.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_breakout_quadrant.add_annotation(text="<b>Strong Breakout & Outperforming</b>", xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_breakout_quadrant.add_annotation(text="<b>Strong Breakout & Underperforming</b>", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_breakout_quadrant.add_annotation(text="<b>Breakdown Risk & Outperforming</b>", xref="paper", yref="paper", x=0.98, y=0.02, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_breakout_quadrant.add_annotation(text="<b>Breakdown Risk & Underperforming</b>", xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_breakout_quadrant.update_traces(textposition='top center', textfont_size=10)
                    fig_breakout_quadrant.update_yaxes(title_text="Breakout Signal", zeroline=False)
                    fig_breakout_quadrant.update_xaxes(title_text="Residual Momentum (vs. BTC)", zeroline=False)
                    fig_breakout_quadrant.update_layout(title_text="Breakout Signal vs. Residual Momentum", height=500, legend_title="Market Phase")
                    st.plotly_chart(fig_breakout_quadrant, use_container_width=True)

                # --- [NEW] EWMAC QUADRANT ---
                st.subheader("EWMAC Trend vs. Momentum Quadrant")
                st.markdown("This chart plots the trend-following signal from a volatility-adjusted EWMAC (Y-axis) against the asset's residual momentum (X-axis).")
                if not df_watchlist.empty:
                    fig_ewmac_quadrant = px.scatter(df_watchlist, x='Residual Momentum', y='EWMAC', text='Token', color='Market Phase', color_discrete_map=phase_colors, hover_data={'Residual Momentum': ':.2f', 'EWMAC': ':.2f'})
                    fig_ewmac_quadrant.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_ewmac_quadrant.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_ewmac_quadrant.add_annotation(text="<b>Bull Trend & Outperforming</b>", xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_ewmac_quadrant.add_annotation(text="<b>Bull Trend & Underperforming</b>", xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_ewmac_quadrant.add_annotation(text="<b>Bear Trend & Outperforming</b>", xref="paper", yref="paper", x=0.98, y=0.02, showarrow=False, align="right", font=dict(color="grey", size=11))
                    fig_ewmac_quadrant.add_annotation(text="<b>Bear Trend & Underperforming</b>", xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False, align="left", font=dict(color="grey", size=11))
                    fig_ewmac_quadrant.update_traces(textposition='top center', textfont_size=10)
                    fig_ewmac_quadrant.update_yaxes(title_text="EWMAC Forecast", zeroline=False)
                    fig_ewmac_quadrant.update_xaxes(title_text="Residual Momentum (vs. BTC)", zeroline=False)
                    fig_ewmac_quadrant.update_layout(title_text="EWMAC Forecast vs. Residual Momentum", height=500, legend_title="Market Phase")
                    st.plotly_chart(fig_ewmac_quadrant, use_container_width=True)
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER (Unchanged)
# ==============================================================================
with tab4:
    st.header("🌊 Wavelet Signal Visualizer")
    st.markdown("This tool denoises price data using wavelets and applies an auto-labeling algorithm to identify potential trend phases. The resulting signals are plotted directly on the price chart.")
    st.sidebar.header("🌊 Wavelet Visualizer Settings")
    symbol_wv = st.sidebar.text_input("Symbol", "ETH/USD", key="wv_symbol")
    timeframe_wv = st.sidebar.selectbox("Timeframe", ['1h', '4h', '1d'], index=2, key="wv_tf")
    limit_wv = st.sidebar.slider("Data Bars", 500, 2000, 1000, key="wv_limit")
    threshold_type_wv = st.sidebar.radio("Threshold Type", ("Volatility-based", "Constant"), key="wv_thresh_type")
    is_constant_disabled = threshold_type_wv != "Constant"
    constant_w_wv = st.sidebar.number_input("Constant Threshold (w)", value=0.015, step=0.001, format="%.4f", key="wv_const_w", disabled=is_constant_disabled)
    
    if st.sidebar.button("Visualize Wavelet Signals", key="run_wv"):
        with st.spinner(f"Generating wavelet signals for {symbol_wv}..."):
            df_wv = fetch_data(symbol_wv, timeframe_wv, limit_wv)
            if df_wv.empty:
                st.error(f"Could not fetch data for {symbol_wv}.")
            else:
                close_prices = df_wv["close"].values
                coeffs = pywt.wavedec(close_prices, 'db4', level=4)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(close_prices)))
                coeffs_thresh = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
                data_denoised = pywt.waverec(coeffs_thresh, 'db4')[:len(close_prices)]
                if threshold_type_wv == "Volatility-based":
                    w_used = rogers_satchell_volatility(df_wv)
                    st.info(f"Using volatility-based threshold (w): {w_used:.4f}")
                else:
                    w_used = constant_w_wv
                    st.info(f"Using constant threshold (w): {w_used:.4f}")
                labels = auto_labeling(data_denoised, w_used)
                df_wv['label'] = labels
                fig_wv = go.Figure()
                fig_wv.add_trace(go.Scatter(x=df_wv.index, y=df_wv['close'], mode='lines', name='Close Price', line=dict(color='gray', width=2)))
                up_signals = df_wv[df_wv['label'] == 1]
                fig_wv.add_trace(go.Scatter(x=up_signals.index, y=up_signals['close'], mode='markers', name='Up Signal', marker=dict(color='deepskyblue', size=7, symbol='circle')))
                down_signals = df_wv[df_wv['label'] == -1]
                fig_wv.add_trace(go.Scatter(x=down_signals.index, y=down_signals['close'], mode='markers', name='Down Signal', marker=dict(color='crimson', size=7, symbol='circle')))
                watermark_text = f"<span style='font-size: 40px;'><b>{symbol_wv}</b></span><br><span style='font-size: 12px; line-height: 0.9em;'>Permutation Research ©</span>"
                fig_wv.add_annotation(text=watermark_text, xref="paper", yref="paper", x=0.05, y=0.98, showarrow=False, font=dict(color="rgba(0, 0, 0, 0.2)"), align="center", xanchor="left", yanchor="top")
                fig_wv.update_layout(title=f'Wavelet Signals on {symbol_wv} Close Price', xaxis_title='Date', yaxis_title='Price (USD)', legend_title='Legend', height=600, paper_bgcolor='rgb(255, 255, 255)', plot_bgcolor='rgb(255, 255, 255)')
                st.plotly_chart(fig_wv, use_container_width=True)
