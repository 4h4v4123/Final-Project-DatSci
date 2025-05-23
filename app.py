import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('C:/Users/Administrator/PycharmProjects/Final Project/Model/lstm_model_raw.h5')

# Load the scaler
with open('C:/Users/Administrator/PycharmProjects/Final Project/scaler_raw.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the dataset
df = pd.read_csv('C:/Users/Administrator/PycharmProjects/Final Project/Market_cleaned_NYA.csv', parse_dates=['Date'], index_col='Date')

# Page title
st.title("LSTM Stock Price NYA Predictor")
st.markdown("""
This app demonstrates how the trained LSTM model predicts stock prices.
You can view:
- Actual vs Predicted prices
- Input your own data to simulate predictions
""")

# Show data
if st.checkbox("Show the First Five Row of the Data"):
    st.write(df.head())

# Feature selection
data = df[['Close', 'Adj Close']].replace(0, 1e-10)

scaled_feature = scaler.transform(data)
scaled_features = pd.DataFrame(scaled_feature, columns=data.columns, index=df.index)

X = []
lookback = 5

# Build sequential input for prediction
for i in range(lookback, len(scaled_features)):
    X.append(scaled_features[i - lookback:i])

X = np.array(X)

# Predict and inverse transform
predicted_scaled = model.predict(X)

predicted = scaler.inverse_transform(predicted_scaled)  # get values back to log-scale

# Build output DataFrame
predicted_df = df.iloc[lookback:].copy()
predicted_df[['Pred_Close', 'Pred_Adj_Close']] = predicted

# Plotting section
st.subheader("Actual vs Predicted Stock Prices")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(predicted_df.index, predicted_df['Close'], label='Actual Close', color='blue')
ax.plot(predicted_df.index, predicted_df['Pred_Close'], label='Predicted Close', color='orange')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Close Price Prediction")
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(predicted_df.index, predicted_df['Adj Close'], label='Actual Adj Close', color='green')
ax2.plot(predicted_df.index, predicted_df['Pred_Adj_Close'], label='Predicted Adjusted Close', color='red')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.set_title("Adjusted Close Price Prediction")
ax2.legend()
st.pyplot(fig2)

# Manual input prediction
st.subheader("Predict with Your Own Input")
st.markdown("Enter the last 5 days of prices to simulate what the model would predict next.")

user_input = []
for i in range(lookback):
    close = st.number_input(f"Day {i+1} - Close Price", value=0.0, format="%.2f", key=f"c{i}")
    adj_close = st.number_input(f"Day {i+1} - Adj Close Price", value=0.0, format="%.2f", key=f"ac{i}")
    user_input.append([close, adj_close])

if st.button("Predict Next Price"):
    input_array = np.array(user_input).reshape(-1, 2)
    scaled_input = scaler.transform(input_array)
    scaled_input = scaled_input.reshape(1, lookback, 2)
    prediction = model.predict(scaled_input)
    predicted_price = scaler.inverse_transform(prediction)

    st.success("Predicted Next Day Prices")
    st.write(f"**Close:** ${predicted_price[0][0]:.2f}")
    st.write(f"**Adjusted Close:** ${predicted_price[0][1]:.2f}")

