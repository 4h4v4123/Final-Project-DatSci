import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

@st.cache_resource
def load_model_once():
    return load_model('tuned_lstm_model.keras')

@st.cache_data
def load_scalers():
    with open('tuned_lstm_scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('tuned_lstm_scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    return scaler_X, scaler_y

@st.cache_data
def load_data():
    return pd.read_csv('Market_cleaned_NYA.csv', parse_dates=['Date'], index_col='Date')

# Load resources
model = load_model_once()
scaler_X, scaler_y = load_scalers()
df = load_data()

# Features and target
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
target = 'Close'

# Scale features
scaled_features = scaler_X.transform(df[features])
scaled_target = scaler_y.transform(df[[target]])

# Create sequences
lookback = 5
X_scaled = np.array([scaled_features[i-lookback:i] for i in range(lookback, len(scaled_features))])
y_scaled = scaled_target[lookback:]

# Predict
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Add predictions to DataFrame
df.loc[df.index[lookback:], 'Predicted_Close'] = y_pred.flatten()

# Streamlit App
st.title("📊 LSTM Market Close Price Predictor")
st.markdown("This app predicts the Close price based on the last 5 days of 5 key features.")

if st.checkbox("🔍 Show Raw Data"):
    st.write(df.tail())

# Plot results
st.subheader("📈 Actual vs Predicted Close Price")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label='Actual Close', color='blue')
ax.plot(df.index, df['Predicted_Close'], label='Predicted Close', color='orange')
ax.set_title("Actual vs Predicted Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Manual input section
st.subheader("🔮 Predict Next Day Close with Your Own Input")
user_input = []
for i in range(lookback):
    st.markdown(f"**Day {i+1}**")
    day_features = []
    for feature in features:
        value = st.number_input(f"{feature}:", value=0.0, format="%.2f", key=f"{feature}_{i}")
        day_features.append(value)
    user_input.append(day_features)

if st.button("Predict Next Day Close"):
    user_input = np.array(user_input)
    scaled_input = scaler_X.transform(user_input).reshape(1, lookback, len(features))
    prediction_scaled = model.predict(scaled_input)
    predicted_close = scaler_y.inverse_transform(prediction_scaled)
    st.success(f"📈 Predicted Close Price for Next Day: ${predicted_close[0][0]:.2f}")
