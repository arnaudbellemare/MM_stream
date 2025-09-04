import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- NEW IMPORT: Import our local, fixed labeler ---
from labeler import AmplitudeBasedLabeler

warnings.filterwarnings('ignore')

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="tr8dr Strategy Backtest")

st.title("🏆 `tr8dr.github.io` Strategy Backtest")
st.markdown("""
This application is a direct implementation of the **`AmplitudeBasedLabeler`** strategy.
- It uses the exact `AmplitudeBasedLabeler` class from the `tseries-patterns` library by including its source code directly in this project, bypassing installation errors.
- It trains a simple ML model on the generated labels and runs a backtest.
""")

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("⚙️ Strategy Configuration")

st.sidebar.subheader("Data Settings")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h', '1d'], index=1)
data_limit = st.sidebar.slider("Number of Data Bars", 1000, 5000, 2000)

# --- UPDATED: Using the exact parameters from the research post ---
st.sidebar.subheader("AmplitudeBasedLabeler Settings")
minamp_bps = st.sidebar.slider("Min Amplitude (bps)", 10, 200, 100)
tinactive_bars = st.sidebar.slider("Trend Inactivity Bars", 5, 50, 10)
# ---

st.sidebar.subheader("Backtest Settings")
train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7)
initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
trade_size = st.sidebar.number_input("Trade Size (in BTC)", value=0.1)


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

@st.cache_data
def fetch_data(symbol, timeframe, limit):
    st.info(f"Fetching {limit} bars of {symbol} {timeframe} data from Kraken...")
    exchange = ccxt.kraken(); ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
    return df

# --- THIS IS THE NEW, ROBUST LABELING FUNCTION USING THE IMPORTED CLASS ---
@st.cache_data
def label_data_with_amplitude_labeler(_df, min_amp, t_inactive):
    st.info("Labeling regimes with AmplitudeBasedLabeler...")
    df_copy = _df.copy()
    labeler = AmplitudeBasedLabeler(minamp=min_amp, Tinactive=t_inactive)
    labels = labeler.transform(df_copy['close'].values)
    df_copy['label'] = labels
    return df_copy
# ---

@st.cache_data
def add_features(df):
    st.info("Engineering features..."); df_copy = df.copy()
    df_copy['return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    df_copy['momentum_24'] = np.log(df_copy['close'] / df_copy['close'].shift(24))
    df_copy['volatility_24'] = df_copy['return'].rolling(window=24).std() * np.sqrt(24 * 365.25)
    df_copy['target'] = df_copy['label'].shift(-8); df_copy.dropna(inplace=True)
    return df_copy

@st.cache_resource
def train_ml_model(_df_train, feature_names):
    st.info("Training Machine Learning model..."); 
    X_train = _df_train[feature_names]; y_train = _df_train['target']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train); 
    return model

@st.cache_data
def run_simple_backtest(_df_backtest, _model, feature_names, cash, size):
    st.info(f"Running backtest...")
    position, equity, trades = 0.0, [cash], []
    for i in range(len(_df_backtest)):
        current_bar = _df_backtest.iloc[i]
        
        # Get prediction from the model trained on the labels
        feature_vector = current_bar[feature_names].values.reshape(1, -1)
        prediction = _model.predict(feature_vector)[0]

        # Exit Logic
        if position > 0 and prediction != 1:
            cash += position * current_bar['open']; trades.append({'t': current_bar.name, 'type': 'EXIT LONG', 'p': current_bar['open']}); position = 0
        elif position < 0 and prediction != -1:
            cash += position * current_bar['open']; trades.append({'t': current_bar.name, 'type': 'EXIT SHORT', 'p': current_bar['open']}); position = 0

        # Entry Logic
        if prediction == 1 and position == 0:
            position += size; cash -= size * current_bar['close']; trades.append({'t': current_bar.name, 'type': 'BUY', 'p': current_bar['close']})
        elif prediction == -1 and position == 0:
            position -= size; cash += size * current_bar['close']; trades.append({'t': current_bar.name, 'type': 'SELL', 'p': current_bar['close']})
            
        equity.append(cash + position * current_bar['close'])
    return pd.DataFrame(trades), pd.Series(equity[1:], index=_df_backtest.index)

# ==============================================================================
# Main App Logic
# ==============================================================================

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Executing full backtest pipeline... Please wait."):
        df_raw = fetch_data(symbol, timeframe, data_limit)
        df_labeled = label_data_with_amplitude_labeler(df_raw, minamp_bps, tinactive_bars)
        df_featured = add_features(df_labeled)
        
        if df_featured.empty:
            st.error("Error: The feature engineering process resulted in an empty dataset. This may happen if the lookback periods are larger than the dataset. Please increase the 'Number of Data Bars'.")
        else:
            split_idx = int(len(df_featured) * train_test_split_ratio); df_train = df_featured.iloc[:split_idx]; df_backtest = df_featured.iloc[split_idx:]
            feature_names = ['momentum_24', 'volatility_24']
            
            if df_train.empty or len(df_train) < 50:
                st.error(f"Error: Not enough training data ({len(df_train)} samples) after splitting. Try adjusting the 'Train/Test Split Ratio' or increasing data size.")
            elif df_backtest.empty:
                st.error(f"Error: Not enough backtesting data ({len(df_backtest)} samples). Try adjusting the 'Train/Test Split Ratio' or increasing data size.")
            else:
                ml_model = train_ml_model(df_train, feature_names)
                trades, equity = run_simple_backtest(df_backtest, ml_model, feature_names, initial_cash, trade_size)

                st.success("✅ Backtest complete!")
                st.header("📊 Performance Summary"); final_equity = equity.iloc[-1]; total_return = (final_equity / initial_cash - 1) * 100
                st.metric("Final Equity (USD)", f"${final_equity:,.2f}"); st.metric("Total Return", f"{total_return:.2f}%"); st.metric("Total Trades (Entries + Exits)", len(trades))
                
                st.header("📈 Charts")
                def plot_results(df_plot, equity_curve, trades):
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Strategy Equity Curve', f'{symbol} Price, Labeled Regimes & Trades'), row_heights=[0.3, 0.7])
                    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name=f'{symbol} Price', line=dict(color='blue')), row=2, col=1)
                    
                    # Plot colored regions for labeled trends
                    up_trends = df_plot[df_plot['label'] == 1]; down_trends = df_plot[df_plot['label'] == -1]
                    for i in range(len(up_trends)): fig.add_vrect(x0=up_trends.index[i], x1=up_trends.index[i] + pd.Timedelta(hours=1), line_width=0, fillcolor="green", opacity=0.1)
                    for i in range(len(down_trends)): fig.add_vrect(x0=down_trends.index[i], x1=down_trends.index[i] + pd.Timedelta(hours=1), line_width=0, fillcolor="red", opacity=0.1)

                    if not trades.empty:
                        buys = trades[trades['type'] == 'BUY']; sells = trades[trades['type'] == 'SELL']; exits = trades[trades['type'].str.contains('EXIT')]
                        fig.add_trace(go.Scatter(x=buys['t'], y=buys['p'], mode='markers', name='Buys', marker=dict(color='lime', symbol='triangle-up', size=10)), row=2, col=1)
                        fig.add_trace(go.Scatter(x=sells['t'], y=sells['p'], mode='markers', name='Sells', marker=dict(color='magenta', symbol='triangle-down', size=10)), row=2, col=1)
                        fig.add_trace(go.Scatter(x=exits['t'], y=exits['p'], mode='markers', name='Exits', marker=dict(color='black', symbol='x', size=8)), row=2, col=1)
                    
                    fig.update_layout(height=800, legend_title='Legend'); return fig
                
                st.plotly_chart(plot_results(df_backtest, equity, trades), use_container_width=True)
                with st.expander("Show Raw Trades Data"): st.dataframe(trades)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Backtest' to begin.")
