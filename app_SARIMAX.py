import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# CONFIG
st.set_page_config(page_title="📈 SARIMAX Forecast Dashboard", layout="wide")

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

df = load_data()
model = load_model()

# SIDEBAR CONTROLS
with st.sidebar:
    st.title("⚙️ Settings")
    forecast_steps = st.slider("Forecast horizon (days)", 1, 30, 7)
    st.markdown("The model will forecast **High** based on **Open** values from this CSV.")
    last_high_value = df["High"].iloc[-1]
    st.number_input("Last known High value", value=float(last_high_value), key="last_high_input")

# MAIN
st.title("📊 SARIMAX Forecasting App - Nikkei 225 (N225)")
st.markdown("Forecasting **High** prices using SARIMAX with **Open** values as exogenous input.")

st.subheader("📋 Preview of Uploaded Data")
st.write(df.head())
st.write(df.tail())

# FORECAST PREP
future_exog = df["Open"].iloc[-forecast_steps:]
future_exog_df = pd.DataFrame({"Open": future_exog})

forecast = model.get_forecast(steps=forecast_steps, exog=future_exog_df)
predicted_diff = forecast.predicted_mean

def inverse_difference(last_ob, diff_series):
    result = []
    current = last_ob
    for diff in diff_series:
        value = current + diff
        result.append(value)
        current = value
    return pd.Series(result)

predicted_high = inverse_difference(last_high_value, predicted_diff)

# FORECAST OUTPUT
st.subheader("🔮 Forecast Results")
forecast_df = pd.DataFrame({
    "Date": pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps),
    "Open": future_exog.values,
    "Predicted High": predicted_high
})

st.dataframe(forecast_df.set_index("Date").style.format({"Open": "{:,.2f}", "Predicted High": "{:,.2f}"}), use_container_width=True)

# PLOTLY CHART
st.subheader("📈 Interactive Forecast Plot")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-50:], y=df["High"].iloc[-50:], mode='lines', name='Historical High'))
fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted High"], mode='lines+markers', name='Forecasted High', line=dict(dash='dash')))
fig.update_layout(title="N225 High Price Forecast", xaxis_title="Date", yaxis_title="High Price", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# EXTRA VISUALS
st.subheader("📊 Market Trend Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**30-day Moving Average of High Prices**")
    ma30 = df["High"].rolling(window=30).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df.index, y=df["High"], mode="lines", name="High"))
    fig_ma.add_trace(go.Scatter(x=df.index, y=ma30, mode="lines", name="30-Day MA"))
    fig_ma.update_layout(template="plotly_white")
    st.plotly_chart(fig_ma, use_container_width=True)

with col2:
    st.markdown("**Correlation Heatmap**")
    fig_corr, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[["Open", "High", "Low", "Close", "Volume"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

# CSV DOWNLOAD
csv = forecast_df.to_csv(index=False).encode()
st.download_button("📥 Download Forecast CSV", csv, "forecast_n225.csv", "text/csv")
