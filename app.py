import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# --- Load model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("untuned_lstm_model.h5")
    scaler = joblib.load("untuned_lstm_model_scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

st.markdown("### 🚀 Forecast Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("RMSE", f"{rmse:.4f}")
col3.metric("Forecast Horizon", f"{forecast_horizon} days")

# --- Streamlit UI --- #
st.title("📈 Market Data LSTM Forecasting")
st.write("Upload market data (CSV) to forecast using a pre-trained LSTM model.")

# --- Sidebar UI ---
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
output_options = ['Adj Close', 'Close', 'Open', 'High', 'Low', 'Volume']

if st.sidebar.checkbox("🧠 Show Model Info"):
    st.subheader("Model Architecture")
    model.summary(print_fn=lambda x: st.text(x))

# --- Sidebar for manual input ---
st.sidebar.header("🔢 Manual Input for Prediction")
selected_output = st.sidebar.selectbox("Select feature to predict", output_options, index=0)
st.sidebar.markdown("Enter the last 5 days of data:")

user_input = []
for i in range(5):
    day = []
    st.sidebar.markdown(f"**Day {i+1}**")
    for feat in features:
        val = st.sidebar.number_input(f"{feat} (Day {i+1})", value=0.0, format="%.2f", key=f"{feat}_{i}")
        day.append(val)
    user_input.append(day)

if st.sidebar.button("Predict Next Day Value"):
    user_input_np = np.array(user_input)
    scaled_input = scaler.transform(user_input_np).reshape(1, 5, len(features))
    prediction = model.predict(scaled_input)
    st.subheader(f"🔮 Predicted Next Day {selected_output}:")
    st.success(f"{selected_output}: {prediction[0][0]:.2f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 6), user_input_np[:, output_options.index(selected_output)], label="Input History")
    ax.scatter(6, prediction[0][0], color='red', label="Predicted Next Value")
    ax.set_xlabel("Day")
    ax.set_ylabel(selected_output)
    ax.set_title(f"{selected_output} - Last 60 Days & Prediction")
    ax.legend()
    st.pyplot(fig)


df = pd.read_csv('Market_cleaned_NYA.csv')
st.subheader("📋 Preview of Uploaded Data")
st.write(df.head())
st.write(df.tail())

# --- Select and scale features ---
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
if not all(col in df.columns for col in features):
    st.error(f"CSV must include these columns: {features}")
else:
    df_features = df[features]
    scaled_data = scaler.transform(df_features)

# --- Create sequences ---
sequence_length = 60
X = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i])
X = np.array(X)
    
st.success(f"✅ Created {X.shape[0]} sequences for prediction.")

# Manual Input Prediction Plot (after st.success() on prediction)
fig2 = go.Figure()

# User input points
fig2.add_trace(go.Scatter(
    x=list(range(1,6)),
    y=user_input_np[:, output_options.index(selected_output)],
    mode='lines+markers',
    name='Input History'
))

# Predicted point
fig2.add_trace(go.Scatter(
    x=[6],
    y=[prediction[0][0]],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Predicted Next Value'
))

fig2.update_layout(
    title=f"📊 {selected_output} - Last 5 Days & Prediction",
    xaxis_title="Day",
    yaxis_title=selected_output,
    template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

# --- Make predictions ---
predictions = model.predict(X)

import plotly.graph_objs as go

# Get actual and predicted values (scale back if needed)
actual = scaled_data[sequence_length:, 3] 
pred = predictions.flatten()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Inverse transform predictions and actuals if necessary (depends on your model setup)
# Assuming they are scaled, so let's just use them directly here
mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100
r2 = r2_score(actual, pred) * 100

st.info(f"📊 **Model Evaluation Metrics:**\n- MAE: {mae:.4f}\n- RMSE: {rmse:.4f}\n- MAPE: {mape:.2f}\n- R2 Score: {r2:.2f}")

# Create interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=actual,
    mode='lines',
    name='Actual (scaled)',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    y=pred,
    mode='lines',
    name='Predicted',
    line=dict(color='orange')
))

fig.update_layout(
    title="📊 Predicted vs Actual Price",
    xaxis_title="Time Step",
    yaxis_title="Scaled Value",
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
