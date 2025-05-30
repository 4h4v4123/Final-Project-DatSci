import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from datetime import datetime, timedelta

# CONFIG
st.set_page_config(
    page_title="📈 SARIMAX Forecast Dashboard | Nikkei 225",
    layout="wide",
    page_icon="📊"
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("Market_cleaned_N225.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

# LOAD MODEL
@st.cache_resource
def load_model():
    return SARIMAXResults.load("sarimax_model.pkl")

# HELPER FUNCTION
def inverse_difference(last_ob, diff_series):
    result = []
    current = last_ob
    for diff in diff_series:
        value = current + diff
        result.append(value)
        current = value
    return pd.Series(result)

# LOAD DATA AND MODEL
df = load_data()
model = load_model()

# SIDEBAR CONTROLS
with st.sidebar:
    st.title("⚙️ Forecast Settings")
    st.markdown("---")
    
    # Forecast horizon
    forecast_steps = st.slider("**Forecast horizon (days)**", 1, 30, 7, help="Select how many days to forecast")
    
    # Last known values
    st.subheader("Last Known Values")
    col1, col2 = st.columns(2)
    with col1:
        last_open_value = st.number_input("Open", value=float(df["Open"].iloc[-1]), format="%.2f")
    with col2:
        last_high_value = st.number_input("High", value=float(df["High"].iloc[-1]), format="%.2f")
    
    st.markdown("---")
    st.subheader("Data Exploration")
    show_full_data = st.checkbox("Show full dataset preview", value=False)
    
    st.markdown("---")
    st.info("💡 The model forecasts **High** prices using SARIMAX with **Open** values as exogenous input.")

# MAIN CONTENT
st.title("📈 SARIMAX Forecasting Dashboard - Nikkei 225 (N225)")
st.markdown("Forecasting High prices using SARIMAX with Open values as exogenous input")
st.markdown("---")

# DATA PREVIEW SECTION
st.subheader("📋 Dataset Preview")
if show_full_data:
    st.dataframe(df, height=300)
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**First 5 Rows**")
        st.dataframe(df.head())
    with col2:
        st.markdown("**Last 5 Rows**")
        st.dataframe(df.tail())
        
    st.markdown(f"**Data Summary**: {len(df)} records from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

st.markdown("---")

# FORECAST PREPARATION
with st.spinner("Generating forecast..."):
    # Create future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps+1)]
    
    # Create future exog DataFrame (using last open value as default)
    future_exog = pd.DataFrame({
        "Open": [last_open_value] * forecast_steps
    }, index=future_dates)
    
    # Generate forecast
    forecast = model.get_forecast(steps=forecast_steps, exog=future_exog)
    predicted_diff = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # Inverse difference transformation
    predicted_high = inverse_difference(last_high_value, predicted_diff)
    conf_int_low = inverse_difference(last_high_value, conf_int.iloc[:, 0])
    conf_int_high = inverse_difference(last_high_value, conf_int.iloc[:, 1])

# FORECAST RESULTS
st.subheader("🔮 Forecast Results")
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Open": future_exog["Open"].values,
    "Predicted High": predicted_high.values,
    "Lower Bound": conf_int_low.values,
    "Upper Bound": conf_int_high.values
})

# Display forecast table
st.dataframe(
    forecast_df.set_index("Date").style.format({
        "Open": "{:,.2f}",
        "Predicted High": "{:,.2f}",
        "Lower Bound": "{:,.2f}",
        "Upper Bound": "{:,.2f}"
    }),
    use_container_width=True
)

# FORECAST VISUALIZATION
st.subheader("📈 Interactive Forecast Visualization")
fig = go.Figure()

# Historical data (last 90 days)
fig.add_trace(go.Scatter(
    x=df.index[-90:], 
    y=df["High"].iloc[-90:], 
    mode='lines',
    name='Historical High',
    line=dict(color='#1f77b4', width=2)
))

# Forecasted data
fig.add_trace(go.Scatter(
    x=forecast_df["Date"], 
    y=forecast_df["Predicted High"], 
    mode='lines+markers',
    name='Forecasted High',
    line=dict(color='#2ca02c', width=3, dash='dash')
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
    y=forecast_df["Upper Bound"].tolist() + forecast_df["Lower Bound"].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(44, 160, 44, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Confidence Interval',
    hoverinfo="skip"
))

# Layout settings
fig.update_layout(
    title='N225 High Price Forecast',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# MARKET ANALYSIS SECTION
st.subheader("📊 Market Trend Analysis")
st.markdown("Explore historical trends and relationships between market indicators")

tab1, tab2, tab3 = st.tabs(["Moving Averages", "Volatility Analysis", "Correlation Matrix"])

with tab1:
    st.markdown("**Moving Average Analysis**")
    ma_period = st.slider("Select moving average period", 7, 90, 30, key="ma_slider")
    
    ma_df = df.copy()
    ma_df[f'{ma_period}-Day MA'] = ma_df['High'].rolling(window=ma_period).mean()
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=ma_df.index, y=ma_df['High'], name='High Price', line=dict(color='#1f77b4')))
    fig_ma.add_trace(go.Scatter(
        x=ma_df.index, 
        y=ma_df[f'{ma_period}-Day MA'], 
        name=f'{ma_period}-Day MA',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig_ma.update_layout(
        title=f'{ma_period}-Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_ma, use_container_width=True)

with tab2:
    st.markdown("**Daily Price Volatility**")
    vol_period = st.slider("Volatility calculation period", 7, 90, 30, key="vol_slider")
    
    vol_df = df.copy()
    vol_df['Daily Change'] = vol_df['High'] - vol_df['Low']
    vol_df['Volatility'] = vol_df['Daily Change'].rolling(window=vol_period).std()
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=vol_df.index, 
        y=vol_df['Volatility'], 
        name='Volatility',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.1)'
    ))
    fig_vol.update_layout(
        title=f'{vol_period}-Day Volatility (Daily Price Range STD)',
        xaxis_title='Date',
        yaxis_title='Volatility',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.markdown("**Feature Correlation Matrix**")
    
    corr_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    
    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    fig_corr.update_layout(
        title='Feature Correlation Matrix',
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# DOWNLOAD SECTION
st.markdown("---")
st.subheader("💾 Download Forecast Results")

col1, col2 = st.columns(2)
with col1:
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Forecast CSV", 
        csv, 
        f"n225_forecast_{datetime.now().strftime('%Y%m%d')}.csv", 
        "text/csv",
        help="Download forecast results in CSV format"
    )

with col2:
    # Create a Plotly figure for export
    export_fig = go.Figure()
    export_fig.add_trace(go.Scatter(x=df.index[-90:], y=df["High"].iloc[-90:], mode='lines', name='Historical High'))
    export_fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted High"], mode='lines', name='Forecasted High'))
    export_fig.add_trace(go.Scatter(
        x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
        y=forecast_df["Upper Bound"].tolist() + forecast_df["Lower Bound"].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    export_fig.update_layout(title='N225 High Price Forecast')
    
    # Download Plotly figure as PNG
    st.download_button(
        "🖼️ Download Forecast Chart", 
        export_fig.to_image(format="png"),
        f"n225_forecast_chart_{datetime.now().strftime('%Y%m%d')}.png",
        "image/png",
        help="Download forecast visualization as PNG image"
    )

# FOOTER
st.markdown("---")
st.caption("SARIMAX Forecast Dashboard | Nikkei 225 (N225) | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))