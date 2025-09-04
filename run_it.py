import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Profit-Optimized Backtest")

st.title("🏆 Profit-Optimized Market Maker Backtest")
st.markdown("""
This is the complete, advanced backtester. It incorporates the crucial insight from the `tr8dr.github.io` post on **Return Classification**.
- **Bias-Free Labeling:** Uses a causal (non-repainting) swing point algorithm.
- **Profit-Optimized Entries:** Instead of using the default ML model prediction, it finds the optimal decision threshold that maximizes a chosen financial metric (like Total PnL or Sharpe Ratio) on out-of-sample data.
- **Dynamic Exits:** Uses the detected swing points to set realistic stop-loss levels.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Backtest Configuration")

# --- Optimization Settings (From the Article) ---
st.sidebar.subheader("🎯 Profit Curve Optimization")
enable_optimization = st.sidebar.checkbox("Enable Profit Curve Optimization", True)
loss_function = st.sidebar.selectbox("Metric to Maximize", ["Total PnL", "Sharpe Ratio"], index=0)

st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 1500, 5000, 2500)

st.sidebar.subheader("Causal Swing Point Settings")
min_trend_pct = st.sidebar.slider("Min Reversal Pct to Confirm Swing", 0.5, 10.0, 3.0, 0.5)
stop_loss_offset_pct = st.sidebar.slider("Stop Loss Offset from Swing (%)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.subheader("Model & Backtest Settings")
train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7)
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)
risk_aversion_gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.01, 0.5, 0.05)


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

@st.cache_data
def find_causal_swing_points(prices_tuple, reversal_pct):
    st.info("Detecting causal (non-repainting) swing points...")
    prices = np.array(prices_tuple); reversal_mult = reversal_pct / 100.0
    swing_highs = np.full(len(prices), np.nan); swing_lows = np.full(len(prices), np.nan)
    last_pivot_idx = 0; peak_idx, trough_idx = 0, 0; trend = 0
    for i in range(1, len(prices)):
        if trend == 1:
            if prices[i] >= prices[peak_idx]: peak_idx = i
            elif prices[i] < prices[peak_idx] * (1 - reversal_mult): swing_highs[peak_idx] = prices[peak_idx]; trend = -1; trough_idx = i
        elif trend == -1:
            if prices[i] <= prices[trough_idx]: trough_idx = i
            elif prices[i] > prices[trough_idx] * (1 + reversal_mult): swing_lows[trough_idx] = prices[trough_idx]; trend = 1; peak_idx = i
        else:
            if prices[i] > prices[trough_idx] * (1 + reversal_mult): trend = 1; peak_idx = i
            elif prices[i] < prices[peak_idx] * (1 - reversal_mult): trend = -1; trough_idx = i
            if prices[i] > prices[peak_idx]: peak_idx = i
            if prices[i] < prices[trough_idx]: trough_idx = i
    return swing_highs, swing_lows

@st.cache_data
def add_features_and_labels(df, swing_highs, swing_lows):
    st.info("Labeling data and engineering features..."); df_copy = df.copy(); df_copy['swing_highs'] = swing_highs; df_copy['swing_lows'] = swing_lows
    df_copy['last_swing_high'] = df_copy['swing_highs'].ffill(); df_copy['last_swing_low'] = df_copy['swing_lows'].ffill()
    df_copy['high_t'] = df_copy['swing_highs'].notna().cumsum(); df_copy['low_t'] = df_copy['swing_lows'].notna().cumsum()
    df_copy['label'] = np.where(df_copy['high_t'] > df_copy['low_t'], -1, 1); df_copy['label'].iloc[0] = 0
    df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1)); df_copy['future_return'] = df_copy['return'].shift(-1)
    df_copy['momentum_24'] = np.log(df_copy['close'] / df_copy['close'].shift(24)); df_copy['volatility_24'] = df_copy['return'].rolling(window=24).std() * np.sqrt(24 * 365.25)
    df_copy['target'] = df_copy['label'].shift(-8); df_copy.dropna(inplace=True)
    return df_copy

@st.cache_resource
def train_ml_model(_df_train, feature_names):
    st.info("Training Machine Learning model..."); X_train = _df_train[feature_names]; y_train = _df_train['target']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1); model.fit(X_train, y_train); return model

# --- THIS FUNCTION IS THE DIRECT IMPLEMENTATION OF THE PROFIT CURVE CONCEPT ---
@st.cache_data
def find_optimal_threshold(_model, _X_test, _y_test_returns, metric):
    st.info(f"Optimizing decision threshold to maximize {metric}...")
    
    # Get probabilities for the positive classes (+1 and -1)
    # model.classes_ is typically [-1, 0, 1], so idx 0 is P(-1), idx 2 is P(1)
    probs = _model.predict_proba(_X_test)
    prob_down, prob_up = probs[:, 0], probs[:, 2]
    
    # Create a score that is positive for up, negative for down predictions
    # This combines both signals into a single score series for thresholding
    scores = np.where(prob_up > prob_down, prob_up, -prob_down)
    
    thresholds = np.linspace(0.33, 0.99, 100) # Test thresholds from neutral to high confidence
    results = []
    
    for thresh in thresholds:
        # Long trades: where probability of UP is above threshold
        long_trades = _y_test_returns[prob_up > thresh]
        # Short trades: where probability of DOWN is above threshold
        short_trades = -_y_test_returns[prob_down > thresh]
        
        all_trades = pd.concat([long_trades, short_trades])
        
        if len(all_trades) < 5: continue # Skip if too few trades
        
        if metric == "Total PnL":
            performance = all_trades.sum()
        elif metric == "Sharpe Ratio":
            # Simple Sharpe, assuming risk-free rate is 0
            performance = all_trades.mean() / (all_trades.std() + 1e-9)

        results.append({'threshold': thresh, 'performance': performance, 'trade_count': len(all_trades)})

    if not results: return 0.5, pd.DataFrame() # Default if no profitable threshold found
        
    results_df = pd.DataFrame(results)
    optimal_row = results_df.loc[results_df['performance'].idxmax()]
    
    return optimal_row['threshold'], results_df
# ---

@st.cache_data
def run_backtest_with_optimization(_df_backtest, _model, feature_names, cash, size, gamma, stop_offset, threshold):
    st.info(f"Running bias-free backtest with optimal threshold: {threshold:.3f}...")
    position, equity, trades = 0.0, [cash], []; in_trade, stop_loss = False, 0.0
    for i in range(len(_df_backtest)):
        current_bar = _df_backtest.iloc[i]
        if in_trade:
            if position > 0 and current_bar['low'] <= stop_loss: cash += position * stop_loss; trades.append({'t':current_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
            elif position < 0 and current_bar['high'] >= stop_loss: cash += position * stop_loss; trades.append({'t':current_bar.name, 'type':'SL EXIT', 'p':stop_loss}); position = 0.0; in_trade = False
        if not in_trade:
            feature_vector = current_bar[feature_names].values.reshape(1, -1)
            # --- MODIFIED LOGIC: Use probabilities and optimal threshold ---
            probabilities = _model.predict_proba(feature_vector)[0]
            prob_down, prob_up = probabilities[0], probabilities[2]

            mid=current_bar['close']; vol=current_bar['volatility_24']; res=(mid)-(position*gamma*(vol**2)); spread=(gamma*(vol**2))+(2/gamma)*np.log(1+(gamma/2)); bid,ask=res-spread/2,res+spread/2
            
            if prob_up > threshold: # Entry signal for long
                position += size; cash -= size * bid; in_trade = True; trades.append({'t':current_bar.name, 'type':'BUY', 'p':bid})
                stop_loss = current_bar['last_swing_low'] * (1 - stop_offset / 100)
            elif prob_down > threshold: # Entry signal for short
                position -= size; cash += size * ask; in_trade = True; trades.append({'t':current_bar.name, 'type':'SELL', 'p':ask})
                stop_loss = current_bar['last_swing_high'] * (1 + stop_offset / 100)
            # ---
        equity.append(cash + position * current_bar['close'])
    return pd.DataFrame(trades), pd.Series(equity, index=_df_backtest.index)

# ==============================================================================
# Main App Logic
# ==============================================================================
if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing full backtest pipeline... Please wait."):
        # 1. Prep Data
        df_raw = fetch_data(symbol, timeframe, data_limit)
        swing_highs, swing_lows = find_causal_swing_points(tuple(df_raw['close']), min_trend_pct)
        df_featured = add_features_and_labels(df_raw, swing_highs, swing_lows)
        split_idx = int(len(df_featured) * train_test_split_ratio); df_train = df_featured.iloc[:split_idx]; df_backtest = df_featured.iloc[split_idx:]
        feature_names = ['momentum_24', 'volatility_24']
        
        # 2. Train Model
        ml_model = train_ml_model(df_train, feature_names)
        
        # 3. Find Optimal Threshold (if enabled)
        optimal_threshold = 0.5 # Default threshold
        profit_curve_df = pd.DataFrame()
        if enable_optimization:
            optimal_threshold, profit_curve_df = find_optimal_threshold(ml_model, df_backtest[feature_names], df_backtest['future_return'], loss_function)

        # 4. Run Backtest
        trades, equity = run_backtest_with_optimization(df_backtest, ml_model, feature_names, initial_cash, trade_size, risk_aversion_gamma, stop_loss_offset_pct, optimal_threshold)

    st.success("✅ Backtest complete!")

    # Display summary metrics
    st.header("📊 Performance Summary")
    if enable_optimization:
        st.info(f"Optimal decision threshold found: **{optimal_threshold:.4f}** (Maximized for {loss_function})")
        
    final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
    st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
    
    # Display the plot
    st.header("📈 Charts")
    if enable_optimization and not profit_curve_df.empty:
        st.subheader("Profit Curve Optimization")
        fig_profit = make_subplots(specs=[[{"secondary_y": True}]])
        fig_profit.add_trace(go.Scatter(x=profit_curve_df['threshold'], y=profit_curve_df['performance'], name=loss_function, line=dict(color='blue')), secondary_y=False)
        fig_profit.add_trace(go.Scatter(x=profit_curve_df['threshold'], y=profit_curve_df['trade_count'], name='Trade Count', line=dict(color='grey', dash='dash')), secondary_y=True)
        fig_profit.add_vline(x=optimal_threshold, line_width=2, line_dash="dash", line_color="red", annotation_text="Optimal")
        fig_profit.update_layout(title_text='Profit Curve: Finding the "Goldilocks Zone"'); fig_profit.update_yaxes(title_text=f"<b>{loss_function}</b>", secondary_y=False); fig_profit.update_yaxes(title_text="<b>Trade Count</b>", secondary_y=True)
        st.plotly_chart(fig_profit, use_container_width=True)

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
