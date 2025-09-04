import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from tseries_patterns import AmplitudeBasedLabeler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import skewnorm, norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="HMM vs. ML Market Maker Backtest")

st.title("📈 HMM vs. ML Market Maker Strategy Backtest")
st.markdown("""
This application runs a comparative backtest between two different models for market regime classification:
1.  **Hidden Markov Model (HMM):** A probabilistic model that uses fitted return distributions.
2.  **Machine Learning (ML):** A RandomForestClassifier trained on engineered features.

Use the sidebar to configure the backtest parameters and click "Run Backtest" to see the results.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Backtest Configuration")

# --- Data Settings ---
st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 1000, 5000, 2000)

# --- Labeler Settings ---
st.sidebar.subheader("Regime Labeler Settings")
min_amplitude_bps = st.sidebar.slider("Min Trend Amplitude (bps)", 50, 500, 150)
t_inactive_bars = st.sidebar.slider("Trend Inactivity Bars", 5, 20, 12)

# --- Model & Backtest Settings ---
st.sidebar.subheader("Model & Backtest Settings")
train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7)
prediction_horizon = st.sidebar.slider("Prediction Horizon (bars)", 1, 24, 8)
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)
max_position = st.sidebar.number_input("Max Position (in BTC)", value=0.3)
risk_aversion_gamma = st.sidebar.slider("Risk Aversion (Gamma)", 0.01, 0.5, 0.05)


# ==============================================================================
# Caching Decorators for Performance
# ==============================================================================
# These decorators store the results of functions so they don't have to be re-run
# every time a widget is changed, making the app much faster.

@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

@st.cache_data
def frac_diff_vect(x_tuple, d): # Streamlit caching requires hashable inputs
    x = np.array(x_tuple)
    n = len(x)
    if n < 2: return x
    x_mean = np.mean(x); x = np.subtract(x, x_mean)
    num = np.cumprod(np.arange(1, n) - d); den = np.cumprod(np.arange(1, n))
    weights = np.divide(num, den); weights = np.insert(weights, 0, 1); weights[1:] = -d * weights[1:]
    ydiff = np.convolve(x, weights, mode='full')[:n]
    return ydiff + x_mean

@st.cache_data
def add_features_and_labels(df, minamp, tinactive, horizon):
    st.info("Labeling data and engineering features...")
    df_copy = df.copy()
    labeler = AmplitudeBasedLabeler(minamp=minamp, Tinactive=tinactive)
    df_copy['label'] = labeler.transform(df_copy['close'].values)
    df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    df_copy['momentum_12'] = np.log(df_copy['close'] / df_copy['close'].shift(12))
    df_copy['momentum_24'] = np.log(df_copy['close'] / df_copy['close'].shift(24))
    df_copy['volatility_24'] = df_copy['return'].rolling(window=24).std() * np.sqrt(24 * 365.25)
    df_copy['frac_diff_price'] = frac_diff_vect(tuple(df_copy['close'].values), d=0.5)
    df_copy['target'] = df_copy['label'].shift(-horizon)
    df_copy.dropna(inplace=True)
    return df_copy

@st.cache_data
def fit_hmm_parameters(df_train):
    st.info("Fitting HMM parameters...")
    up = df_train[df_train['label'] == 1]['return'].dropna(); neutral = df_train[df_train['label'] == 0]['return'].dropna(); down = df_train[df_train['label'] == -1]['return'].dropna()
    dist_params = {"up": skewnorm.fit(up), "neutral": norm.fit(neutral), "down": skewnorm.fit(down)}
    labels = df_train['label'].values
    state_map = {-1: 0, 0: 1, 1: 2}; y_true = [state_map[s] for s in labels[:-1]]; y_pred = [state_map[s] for s in labels[1:]]
    counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    trans_matrix = counts / counts.sum(axis=1, keepdims=True)
    return {"dists": dist_params, "transmat": trans_matrix}

@st.cache_data(allow_output_mutation=True)
def train_ml_model(df_train, feature_names):
    st.info("Training Machine Learning model...")
    X_train = df_train[feature_names]; y_train = df_train['target']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

@st.cache_data
def run_comparative_backtest(_df_backtest, hmm_params, _ml_model, feature_names, cash, size, max_pos, gamma):
    class OnlineRegimeClassifierHMM:
        def __init__(self, trans_matrix, dist_params):
            self.states = [-1, 0, 1]; self.trans_matrix = trans_matrix
            self.dists = {-1: skewnorm(*dist_params["down"]), 0: norm(*dist_params["neutral"]), 1: skewnorm(*dist_params["up"])}
            self.state_probs = np.array([1/3, 1/3, 1/3])
        def get_current_state(self, x):
            emission_probs = np.array([self.dists[s].pdf(x) for s in self.states]) + 1e-9
            predicted_probs = self.state_probs @ self.trans_matrix; new_state_probs = predicted_probs * emission_probs
            self.state_probs = new_state_probs / np.sum(new_state_probs); return self.states[np.argmax(self.state_probs)]

    st.info("Running comparative backtest simulation...")
    hmm_classifier = OnlineRegimeClassifierHMM(hmm_params['transmat'], hmm_params['dists'])
    cash_hmm, pos_hmm, eq_hmm, tr_hmm = cash, 0.0, [cash], []
    cash_ml, pos_ml, eq_ml, tr_ml = cash, 0.0, [cash], []

    for i in range(len(_df_backtest) - 1):
        curr, next_b = _df_backtest.iloc[i], _df_backtest.iloc[i + 1]
        
        # HMM
        pred_hmm = hmm_classifier.get_current_state(curr['return'])
        if pred_hmm != 0:
            mid = curr['close']; vol = curr['volatility_24']; ofi = (curr['close']-curr['open'])/(curr['high']-curr['low']+1e-9)
            res = (mid+ofi*5)-(pos_hmm*gamma*(vol**2)); spread = (gamma*(vol**2))+(2/gamma)*np.log(1+(gamma/2))
            bid, ask = res-spread/2, res+spread/2
            if pos_hmm<max_pos and next_b['low']<=bid: pos_hmm+=size; cash_hmm-=size*bid; tr_hmm.append({'t':next_b.name,'type':'BUY','p':bid})
            elif pos_hmm>-max_pos and next_b['high']>=ask: pos_hmm-=size; cash_hmm+=size*ask; tr_hmm.append({'t':next_b.name,'type':'SELL','p':ask})
        
        # ML
        fv = curr[feature_names].values.reshape(1, -1)
        pred_ml = _ml_model.predict(fv)[0]
        if pred_ml != 0:
            mid = curr['close']; vol = curr['volatility_24']; ofi = (curr['close']-curr['open'])/(curr['high']-curr['low']+1e-9)
            res = (mid+ofi*5)-(pos_ml*gamma*(vol**2)); spread = (gamma*(vol**2))+(2/gamma)*np.log(1+(gamma/2))
            bid, ask = res-spread/2, res+spread/2
            if pos_ml<max_pos and next_b['low']<=bid: pos_ml+=size; cash_ml-=size*bid; tr_ml.append({'t':next_b.name,'type':'BUY','p':bid})
            elif pos_ml>-max_pos and next_b['high']>=ask: pos_ml-=size; cash_ml+=size*ask; tr_ml.append({'t':next_b.name,'type':'SELL','p':ask})

        eq_hmm.append(cash_hmm + pos_hmm * curr['close'])
        eq_ml.append(cash_ml + pos_ml * curr['close'])

    return (pd.DataFrame(tr_hmm), pd.Series(eq_hmm, index=_df_backtest.index)), (pd.DataFrame(tr_ml), pd.Series(eq_ml, index=_df_backtest.index))

def plot_results(df_plot, hmm_results, ml_results):
    trades_hmm, equity_hmm = hmm_results; trades_ml, equity_ml = ml_results
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Strategy Performance Comparison', f'{symbol} Price & Trades'), row_heights=[0.4, 0.6])
    fig.add_trace(go.Scatter(x=equity_hmm.index, y=equity_hmm, mode='lines', name='HMM Equity', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=equity_ml.index, y=equity_ml, mode='lines', name='ML Equity', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=2, col=1)
    if not trades_hmm.empty:
        fig.add_trace(go.Scatter(x=trades_hmm['t'], y=trades_hmm['p'], mode='markers', name='HMM Buys', marker=dict(color='cyan', symbol='triangle-up', size=12, line=dict(width=1, color='DarkSlateGrey'))), row=2, col=1)
        fig.add_trace(go.Scatter(x=trades_hmm[trades_hmm['type'] == 'SELL']['t'], y=trades_hmm[trades_hmm['type'] == 'SELL']['p'], mode='markers', name='HMM Sells', marker=dict(color='magenta', symbol='triangle-down', size=12, line=dict(width=1, color='DarkSlateGrey'))), row=2, col=1)
    if not trades_ml.empty:
        fig.add_trace(go.Scatter(x=trades_ml[trades_ml['type'] == 'BUY']['t'], y=trades_ml[trades_ml['type'] == 'BUY']['p'], mode='markers', name='ML Buys', marker=dict(color='green', symbol='circle', size=8)), row=2, col=1)
        fig.add_trace(go.Scatter(x=trades_ml[trades_ml['type'] == 'SELL']['t'], y=trades_ml[trades_ml['type'] == 'SELL']['p'], mode='markers', name='ML Sells', marker=dict(color='red', symbol='circle', size=8)), row=2, col=1)
    fig.update_layout(height=800, legend_title='Legend')
    return fig

# ==============================================================================
# Main App Logic
# ==============================================================================

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing full backtest pipeline... Please wait."):
        # 1. Fetch and process data (cached)
        df_raw = fetch_data(symbol, timeframe, data_limit)
        df_featured = add_features_and_labels(df_raw, min_amplitude_bps, t_inactive_bars, prediction_horizon)

        # 2. Split data
        split_idx = int(len(df_featured) * train_test_split_ratio)
        df_train = df_featured.iloc[:split_idx]
        df_backtest = df_featured.iloc[split_idx:]
        feature_names = ['momentum_12', 'momentum_24', 'volatility_24', 'frac_diff_price']

        # 3. Fit/Train models (cached)
        hmm_params = fit_hmm_parameters(df_train)
        ml_model = train_ml_model(df_train, feature_names)

        # 4. Run backtest (cached)
        hmm_results, ml_results = run_comparative_backtest(df_backtest, hmm_params, ml_model, feature_names,
                                                           initial_cash, trade_size, max_position, risk_aversion_gamma)

    st.success("✅ Backtest complete!")

    # Display summary metrics
    st.header("📊 Performance Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("HMM Strategy Results")
        final_equity_hmm = hmm_results[1].iloc[-1]
        st.metric("Final Equity (USD)", f"${final_equity_hmm:,.2f}")
        st.metric("Total Return", f"{(final_equity_hmm / initial_cash - 1) * 100:.2f}%")
        st.metric("Total Trades", len(hmm_results[0]))

    with col2:
        st.subheader("ML Strategy Results")
        final_equity_ml = ml_results[1].iloc[-1]
        st.metric("Final Equity (USD)", f"${final_equity_ml:,.2f}")
        st.metric("Total Return", f"{(final_equity_ml / initial_cash - 1) * 100:.2f}%")
        st.metric("Total Trades", len(ml_results[0]))
    
    # Display the plot
    st.header("📈 Charts")
    fig = plot_results(df_backtest, hmm_results, ml_results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw trade data in expanders
    with st.expander("Show HMM Trades Data"):
        st.dataframe(hmm_results[0])
    with st.expander("Show ML Trades Data"):
        st.dataframe(ml_results[0])

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
