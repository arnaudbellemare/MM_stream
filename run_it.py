import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Quantitative Research Dashboard")

# --- NEW: Import and apply the custom 'Tourney' font ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tourney:ital,wght@1,100&display=swap');

.dashboard-title {
    font-family: 'Tourney', cursive;
    font-weight: 100;
    font-style: italic;
    font-size: 42px;
    text-align: center;
    padding-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Replace the original st.title with our new styled title
st.markdown('<h1 class="dashboard-title">QUANTITATIVE TRADING RESEARCH DASHBOARD</h1>', unsafe_allow_html=True)
# --- END of new code ---

st.markdown("<div style='text-align: center;'>An interactive dashboard combining <strong>AI-driven predictions</strong> with <strong>in-depth statistical analysis</strong> for a comprehensive market view.</div>", unsafe_allow_html=True)


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

# ==============================================================================
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
    'C98', 'BODEN', 'HDX', 'MSOL', 'REP', 'ANLOG', 'RLUSD', 'USDT','EUROP'}

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

def get_triple_barrier_labels(close, lookahead_periods=5, vol_mult=1.5):
    returns = close.pct_change(); volatility = returns.rolling(20).std().fillna(method='bfill')
    labels = pd.Series(0, index=close.index)
    for i in range(len(close) - lookahead_periods):
        entry = close.iloc[i]; vol = volatility.iloc[i]
        if vol == 0: continue
        tp = entry * (1 + vol_mult * vol); sl = entry * (1 - vol_mult * vol)
        future = close.iloc[i+1 : i+1+lookahead_periods]
        if (future >= tp).any(): labels.iloc[i] = 1
        elif (future <= sl).any(): labels.iloc[i] = -1
    return labels

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

def generate_residual_momentum_factor(asset_prices, market_prices, window=30):
    asset_returns = np.log(asset_prices / asset_prices.shift(1))
    market_returns = np.log(market_prices / market_prices.shift(1))
    rolling_cov = asset_returns.rolling(window=window).cov(market_returns)
    rolling_var = market_returns.rolling(window=window).var()
    beta = rolling_cov / rolling_var
    residual = asset_returns - (beta * market_returns)
    return (residual-residual.rolling(window).mean())/residual.rolling(window).std()

# ==============================================================================
# TAB 3: COMPREHENSIVE WATCHLIST
# ==============================================================================
with tab3:
    st.header("📈 Comprehensive Watchlist")
    st.markdown("""
    This powerful tool generates a unified watchlist by combining **AI-driven predictions** with **robust statistical analysis**.
    - **MLP Signal & Confidence:** A unique neural network is trained for each asset to predict the next market move and its confidence.
    - **Statistical Metrics:** Includes rule-based momentum phase, wavelet-based performance, and residual momentum.
    """)
    st.sidebar.header("📈 Watchlist Configuration")
    min_volume_wl = st.sidebar.number_input("Minimum 24h Quote Volume", value=250000, key="wl_min_vol")
    data_limit_wl = st.sidebar.slider("Data Bars for Analysis", 500, 2000, 1500, key="wl_limit")

    @st.cache_data(ttl=3600 * 2) # Cache results for 2 hours
    def generate_comprehensive_watchlist(symbols, timeframe, limit):
        st.info(f"Starting comprehensive analysis for {len(symbols)} tokens...")
        results = []
        progress_bar = st.progress(0, text="Initializing analysis...")

        market_df = fetch_data('BTC/USD', timeframe, limit)
        if market_df.empty:
            st.error("Could not fetch market data (BTC/USD). Cannot proceed."); return pd.DataFrame()

        for i, symbol in enumerate(symbols):
            try:
                df = fetch_data(symbol, timeframe, limit)
                if df.empty or len(df) < 100: continue
                
                df_mlp = df.copy()
                df_mlp['ret_fast'] = df_mlp['close'].pct_change(7)
                df_mlp['ret_slow'] = df_mlp['close'].pct_change(30)
                df_mlp['volatility'] = df_mlp['close'].pct_change().rolling(20).std()
                df_mlp['adx'] = get_adx(df_mlp['high'], df_mlp['low'], df_mlp['close'], 14)
                df_mlp['volume_z'] = (df_mlp['volume']-df_mlp['volume'].rolling(30).mean())/df_mlp['volume'].rolling(30).std()
                
                features_df = df_mlp[['ret_fast', 'ret_slow', 'volatility', 'adx', 'volume_z']].dropna()
                if len(features_df) < 50: continue

                labels = get_triple_barrier_labels(df_mlp['close']).loc[features_df.index]
                
                pipeline = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42, early_stopping=True))
                pipeline.fit(features_df, labels)
                
                latest_features = features_df.iloc[-1:]
                pred_code = pipeline.predict(latest_features)[0]
                pred_proba = pipeline.predict_proba(latest_features)[0].max()
                signal_map = {1: "Buy", -1: "Sell", 0: "Hold"}
                mlp_signal = signal_map.get(pred_code, "Hold")
                confidence = pred_proba
                
                fast_ret = df['close'].iloc[-1] / df['close'].iloc[-8] - 1 if len(df) > 8 else 0
                slow_ret = df['close'].iloc[-1] / df['close'].iloc[-31] - 1 if len(df) > 31 else 0
                W_FAST = 1 if fast_ret >= 0 else -1; W_SLOW = 1 if slow_ret >= 0 else -1
                if W_SLOW==1 and W_FAST==1: market_phase="Bull"
                elif W_SLOW==-1 and W_FAST==-1: market_phase="Bear"
                elif W_SLOW==1 and W_FAST==-1: market_phase="Correction"
                else: market_phase="Rebound"

                close_prices = df["close"].values
                coeffs = pywt.wavedec(close_prices, 'db4', level=4); sigma = np.median(np.abs(coeffs[-1]))/0.6745
                uthresh = sigma * np.sqrt(2*np.log(len(close_prices)))
                coeffs_thresh = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
                data_denoised = pywt.waverec(coeffs_thresh, 'db4')[:len(close_prices)]
                
                w = rogers_satchell_volatility(df); wv_labels = auto_labeling(data_denoised, w)
                df['wv_label'] = wv_labels
                df['log_ret'] = np.log(df['close']/df['close'].shift(1))
                df['strat_ret'] = df['wv_label'].shift(1) * df['log_ret']
                net_bps = df['strat_ret'].sum() * 10000
                bull_bear_bias = df['wv_label'].mean()
                gt = np.sign(df['close'].shift(-1) - df['close']).fillna(0)
                accuracy = accuracy_score(gt, df['wv_label'])
                
                res_mom = generate_residual_momentum_factor(df['close'], market_df['close'])
                res_mom_score = res_mom.iloc[-1] if not res_mom.empty and pd.notna(res_mom.iloc[-1]) else 0.0

                results.append({
                    'Token': symbol, 'MLP Signal': mlp_signal, 'Confidence': confidence, 'Market Phase': market_phase,
                    'Bull/Bear Bias': bull_bear_bias, 'Net BPS': net_bps, 'Wavelet Accuracy': accuracy, 'Residual Momentum': res_mom_score,
                })
            except Exception: continue
            finally: progress_bar.progress((i + 1) / len(symbols), text=f"Analyzed {symbol}...")
        
        progress_bar.empty(); return pd.DataFrame(results)

    if st.sidebar.button("📈 Run Comprehensive Analysis", key="run_wl"):
        watchlist_symbols = get_filtered_tickers(min_volume_wl)
        
        if not watchlist_symbols:
            st.error("No tickers met the filter criteria. Watchlist is empty.")
        else:
            df_watchlist = generate_comprehensive_watchlist(watchlist_symbols, '1d', data_limit_wl)
            
            if df_watchlist.empty:
                st.warning("Analysis complete, but no data could be generated for the watchlist.")
            else:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("Comprehensive Market Watchlist")
                    df_display = df_watchlist.sort_values(by='Confidence', ascending=False).reset_index(drop=True)
                    
                    df_display['Confidence'] = df_display['Confidence'].map('{:.1%}'.format)
                    df_display['Bull/Bear Bias'] = df_display['Bull/Bear Bias'].map('{:+.2%}'.format)
                    df_display['Net BPS'] = df_display['Net BPS'].map('{:,.0f}'.format)
                    df_display['Wavelet Accuracy'] = df_display['Wavelet Accuracy'].map('{:.1%}'.format)
                    df_display['Residual Momentum'] = df_display['Residual Momentum'].map('{:+.2f}'.format)
                    
                    column_order = [
                        'Token', 'MLP Signal', 'Confidence', 'Market Phase', 'Bull/Bear Bias',
                        'Net BPS', 'Wavelet Accuracy', 'Residual Momentum'
                    ]
                    st.dataframe(df_display[column_order], use_container_width=True, hide_index=True)

                with col2:
                    st.subheader("Market Sentiment")
                    st.markdown("<h5 style='text-align: center;'>Market Phase Distribution</h5>", unsafe_allow_html=True)
                    
                    phase_counts = df_watchlist['Market Phase'].value_counts()
                    
                    phase_colors = {
                        'Bull': 'mediumseagreen', 'Bear': 'crimson',
                        'Correction': 'orange', 'Rebound': 'deepskyblue'
                    }

                    fig_donut = px.pie(
                        values=phase_counts.values, 
                        names=phase_counts.index,
                        hole=0.5,
                        color=phase_counts.index,
                        color_discrete_map=phase_colors
                    )
                    fig_donut.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value')
                    
                    # --- UPDATED ANNOTATION WITH TOURNEY FONT ---
                    fig_donut.add_annotation(
                        text="""<span style="font-family: 'Tourney', sans-serif; font-weight: 100; font-style: italic; font-size: 18px;">
                                PERMUTATION<br>RESEARCH
                                </span>""",
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(
                            color="black"
                        ),
                        align="center"
                    )
                    # --- END OF UPDATE ---

                    fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=20, l=20, r=20))
                    st.plotly_chart(fig_donut, use_container_width=True)
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
# ==============================================================================
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
# ==============================================================================
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
# ==============================================================================
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
# ==============================================================================
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
# ==============================================================================
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
# ==============================================================================
# ==============================================================================
# TAB 4: WAVELET SIGNAL VISUALIZER
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
                # Denoising and Labeling logic remains the same...
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

                # Plotting
                fig_wv = go.Figure()
                fig_wv.add_trace(go.Scatter(x=df_wv.index, y=df_wv['close'], mode='lines', name='Close Price', line=dict(color='gray', width=2)))
                
                up_signals = df_wv[df_wv['label'] == 1]
                fig_wv.add_trace(go.Scatter(x=up_signals.index, y=up_signals['close'], mode='markers', name='Up Signal', marker=dict(color='deepskyblue', size=7, symbol='circle')))
                
                down_signals = df_wv[df_wv['label'] == -1]
                fig_wv.add_trace(go.Scatter(x=down_signals.index, y=down_signals['close'], mode='markers', name='Down Signal', marker=dict(color='crimson', size=7, symbol='circle')))

                watermark_text = (
                    f"<span style='font-size: 40px;'><b>{symbol_wv}</b></span><br>"
                    f"<span style='font-size: 12px; line-height: 0.9em;'>Permutation Research ©</span>"
                )
                
                fig_wv.add_annotation(
                    text=watermark_text, xref="paper", yref="paper",
                    x=0.05, y=0.98, showarrow=False,
                    font=dict(color="rgba(0, 0, 0, 0.2)"),
                    align="center", xanchor="left", yanchor="top"
                )

                # --- NEW: Updated layout with background color ---
                fig_wv.update_layout(
                    title=f'Wavelet Signals on {symbol_wv} Close Price',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    legend_title='Legend',
                    height=600,
                    paper_bgcolor='rgb(255, 255, 255)', # Background for the entire figure area
                    plot_bgcolor='rgb(255, 255, 255)'  # Background for the plotting area
                )
                # --- END of new code ---
                
                st.plotly_chart(fig_wv, use_container_width=True)
