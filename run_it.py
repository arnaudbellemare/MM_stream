import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

st.title(" QUANTITATIVE TRADING RESEARCH DASHBOARD")
st.markdown("An interactive dashboard for backtesting strategies and analyzing advanced market signals. This version includes critical bug fixes for caching and data indexing.")

# ==============================================================================
# Helper Functions (used across tabs)
# ==============================================================================
@st.cache_data
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.kraken()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if len(ohlcv) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['stamp'] = df.index
        return df
    except Exception:
        return pd.DataFrame()


def ema(data, window):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

# ==============================================================================
# TAB STRUCTURE
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["🏆 Hawkes Strategy Backtester", "🔬 SG Swing Analysis", "🌊 Wavelet Auto-Labeling"])


# ==============================================================================
# TAB 1: HAWKES STRATEGY BACKTESTER
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
# TAB 3: WAVELET AUTO-LABELING
# ==============================================================================
with tab3:
    st.header("Wavelet Auto-Labeling & Performance Metrics")
    st.markdown("This section implements the advanced labeling technique using wavelet denoising and evaluates its performance against a simple ground truth.")
    st.sidebar.subheader("Wavelet Labeling Settings")
    symbol_wl = st.sidebar.text_input("Symbol", "BTC/USD", key="wl_symbol")
    threshold_type_wl = st.sidebar.radio("Threshold Type", ("Volatility-based", "Constant"), key="wl_thresh_type")
    constant_w_wl = st.sidebar.number_input("Constant Threshold (w)", value=0.01, step=0.001, format="%.4f", key="wl_const_w")

    STABLECOINS = {
        'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD', 'USDK', 'UST', 'SUSD', 'FRAX', 'LUSD', 'MIM', 'USDQ',
        'TBTC', 'WBTC', 'EUL', 'EUR', 'EURT', 'USDS', 'USTS', 'USTC', 'USDR', 'PYUSD', 'EURR', 'GBP', 'AUD', 'EURQ',
        'T', 'USDG', 'WAXL', 'IDEX', 'FIS', 'CSM', 'MV', 'POWR', 'ATLAS', 'XCN', 'BOBA', 'OXY', 'BNC', 'POLIS', 'AIR',
        'C98', 'BODEN', 'HDX', 'MSOL', 'REP', 'ANLOG', 'RLUSD', 'USDT','EUROP'
    }
    
    # --- FINAL, CORRECTED TICKER FETCHING AND FILTERING LOGIC ---
    @st.cache_data(ttl=3600) # Cache for 1 hour
    def get_filtered_tickers(min_quote_volume=100000):
        st.info("Fetching all tickers from Kraken to filter by live 24h volume...")
        try:
            exchange = ccxt.kraken()
            tickers = exchange.fetch_tickers()
            markets = exchange.load_markets()
            
            filtered_symbols = []
            for symbol, ticker in tickers.items():
                # --- APPLY NEW, STRICT FILTERS ---
                # 1. Symbol MUST end with /USD. No exceptions.
                if not symbol.endswith('/USD'):
                    continue
                
                # Check for necessary data points
                if 'quoteVolume' not in ticker or ticker['quoteVolume'] is None:
                    continue
                if symbol not in markets:
                    continue

                # 2. Live 24h quote volume must be above the threshold.
                if ticker['quoteVolume'] < min_quote_volume:
                    continue

                # 3. Base currency must NOT be in the exclusion list.
                base_currency = markets[symbol].get('base')
                if base_currency in STABLECOINS:
                    continue

                filtered_symbols.append(symbol)
            
            st.success(f"Found {len(filtered_symbols)} tickers ending in /USD with >{min_quote_volume:,} volume.")
            return filtered_symbols
            
        except Exception as e:
            st.error(f"Failed to fetch or filter tickers from Kraken: {e}")
            return []

    @st.cache_data
    def auto_labeling(data_tuple, timestamp_tuple, w):
        data_list = np.array(data_tuple); timestamps = pd.Series(timestamp_tuple)
        labels = np.zeros(len(data_list)); FP = data_list[0]; x_H = data_list[0]; HT = timestamps[0]; x_L = data_list[0]; LT = timestamps[0]; Cid = 0; FP_N = 0
        for i in range(len(data_list)):
            if data_list[i] > FP + FP * w: x_H = data_list[i]; HT = timestamps[i]; FP_N = i; Cid = 1; break
            if data_list[i] < FP - FP * w: x_L = data_list[i]; LT = timestamps[i]; FP_N = i; Cid = -1; break
        for i in range(FP_N, len(data_list)):
            if Cid > 0:
                if data_list[i] > x_H: x_H = data_list[i]; HT = timestamps[i]
                if data_list[i] < x_H - x_H * w and LT < HT:
                    mask = ((timestamps > LT) & (timestamps <= HT)).values; labels[mask] = 1
                    x_L = data_list[i]; LT = timestamps[i]; Cid = -1
            elif Cid < 0:
                if data_list[i] < x_L: x_L = data_list[i]; LT = timestamps[i]
                if data_list[i] > x_L + x_L * w and HT <= LT:
                    mask = ((timestamps > HT) & (timestamps <= LT)).values; labels[mask] = -1
                    x_H = data_list[i]; HT = timestamps[i]; Cid = 1
        labels = np.where(labels == 0, Cid, labels); return labels

    def rogers_satchell_volatility(data):
        log_ho = np.log(data["high"] / data["open"]); log_lo = np.log(data["low"] / data["open"]); log_co = np.log(data["close"] / data["open"])
        return np.sqrt(np.mean(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)))

    @st.cache_data
    def generate_watchlist(symbols, timeframe, limit):
        st.info(f"Analyzing {len(symbols)} tokens for watchlist...")
        watchlist_results = []
        progress_bar = st.progress(0, text="Initializing watchlist analysis...")
        
        for i, symbol in enumerate(symbols):
            try:
                df = fetch_data(symbol, timeframe, limit)
                if df.empty or len(df) < 50: continue

                data_train = df["close"].values; timestamps_train = df.index
                coeffs = pywt.wavedec(data_train, 'db4', level=4); sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(data_train))); coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
                data_denoised = pywt.waverec(coeffs_thresh, 'db4')
                
                min_len = min(len(data_denoised), len(timestamps_train)); data_denoised = data_denoised[:min_len]; df = df.iloc[:min_len].copy()

                w = rogers_satchell_volatility(df); labels = auto_labeling(tuple(data_denoised), tuple(df.index), w)
                df['label'] = labels

                df['log_return'] = np.log(df['close'] / df['close'].shift(1)); df['strategy_return'] = df['label'].shift(1) * df['log_return']
                total_bps = df['strategy_return'].sum() * 10000

                bull_dots = (df['label'] == 1).sum(); bear_dots = (df['label'] == -1).sum(); total_dots = bull_dots + bear_dots
                bull_bear_bias = (bull_dots - bear_dots) / total_dots if total_dots > 0 else 0
                
                gt = np.sign(df['close'].shift(-1) - df['close']).fillna(0); accuracy = accuracy_score(gt, df['label'])
                
                watchlist_results.append({'Token': symbol, 'Net BPS': total_bps, 'Bull/Bear Bias': f"{bull_bear_bias:.2%}",'Accuracy': f"{accuracy:.2%}"})

            except Exception as e:
                st.warning(f"Could not process {symbol}. Error: {e}")
            
            progress_bar.progress((i + 1) / len(symbols), text=f"Analyzing {symbol}...")
        
        progress_bar.empty()
        if not watchlist_results: return pd.DataFrame()

        df_watchlist = pd.DataFrame(watchlist_results)
        df_watchlist = df_watchlist.sort_values(by='Net BPS', ascending=False).reset_index(drop=True)
        df_watchlist['Net BPS'] = df_watchlist['Net BPS'].map('{:,.2f}'.format)
        
        return df_watchlist


    if st.sidebar.button("🌊 Run Wavelet Analysis", key="run_wl"):
        df_wl = fetch_data(symbol_wl, '1h', 1000)
        if df_wl.empty:
            st.error(f"Could not fetch data for {symbol_wl}. Cannot perform analysis.")
        else:
            data_train = df_wl["close"].values; timestamps_train = df_wl.index
            st.info("Denoising data with Wavelets..."); coeffs = pywt.wavedec(data_train, 'db4', level=4); sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(data_train))); coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]; data_train_denoised = pywt.waverec(coeffs_thresh, 'db4')
            min_len = min(len(data_train_denoised), len(timestamps_train)); data_train_denoised = data_train_denoised[:min_len]; timestamps_train = timestamps_train[:min_len]; df_wl = df_wl.iloc[:min_len].copy()
            df_wl['denoised_close'] = data_train_denoised

            if threshold_type_wl == "Volatility-based": w_used = rogers_satchell_volatility(df_wl); st.write(f"**Volatility-based threshold (w):** `{w_used:.4f}`")
            else: w_used = constant_w_wl; st.write(f"**Using constant threshold (w):** `{w_used:.4f}`")

            st.info("Auto-labeling denoised data...")
            labels_wavelet = auto_labeling(tuple(data_train_denoised), tuple(timestamps_train), w_used)
            df_wl['label'] = labels_wavelet

            gt_labels = np.sign(df_wl['close'].shift(-1) - df_wl['close']).fillna(0)
            accuracy = accuracy_score(gt_labels, labels_wavelet); precision = precision_score(gt_labels, labels_wavelet, average='weighted', zero_division=0); recall = recall_score(gt_labels, labels_wavelet, average='weighted', zero_division=0)

            st.header("Performance Metrics")

            col1, col2, col3, col4 = st.columns([1, 1, 1, 2.5])
            col1.metric("Accuracy", f"{accuracy:.2%}"); col2.metric("Precision", f"{precision:.2%}"); col3.metric("Recall", f"{recall:.2%}")

            with col4:
                st.subheader("🏆 Dynamic USD Watchlist")
                st.markdown("Auto-filtered tokens ranked by **Net BPS** (strongest directional structure).")
                
                watchlist_symbols = get_filtered_tickers(min_quote_volume=100000)
                if watchlist_symbols:
                    df_watchlist = generate_watchlist(watchlist_symbols, '1h', 1000)
                    if not df_watchlist.empty:
                        st.dataframe(df_watchlist, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Analysis complete, but no data to display for the watchlist.")
                else:
                    st.error("No tickers met the filter criteria. Watchlist is empty.")
            
            st.header("Charts")
            fig_wl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Denoised Data & Labels', f'Original Price for {symbol_wl}'))

            fig_wl.add_trace(go.Scatter(x=df_wl.index, y=df_wl['denoised_close'], name='Wavelet Denoised Price', line=dict(color='orange')), row=1, col=1)
            up = df_wl[df_wl['label']==1]
            down = df_wl[df_wl['label']==-1]
            fig_wl.add_trace(go.Scatter(x=up.index, y=up['denoised_close'], mode='markers', name='Up Label', marker=dict(color='deepskyblue', symbol='circle', size=5)), row=1, col=1)
            fig_wl.add_trace(go.Scatter(x=down.index, y=down['denoised_close'], mode='markers', name='Down Label', marker=dict(color='red', symbol='circle', size=5)), row=1, col=1)

            fig_wl.add_trace(go.Scatter(x=df_wl.index, y=df_wl['close'], name='Original Price', line=dict(color='gray')), row=2, col=1)

            fig_wl.add_trace(go.Scatter(
                x=up.index, y=up['close'], mode='markers', name='Up Label (on Price)',
                marker=dict(color='deepskyblue', symbol='circle', size=5), showlegend=False
            ), row=2, col=1)

            fig_wl.add_trace(go.Scatter(
                x=down.index, y=down['close'], mode='markers', name='Down Label (on Price)',
                marker=dict(color='red', symbol='circle', size=5), showlegend=False
            ), row=2, col=1)

            fig_wl.update_layout(height=800, legend_title_text='Legend')
            st.plotly_chart(fig_wl, use_container_width=True)
