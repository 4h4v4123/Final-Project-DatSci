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

# --- Streamlit UI --- #
st.title("📈 Market Data LSTM Forecasting")
st.write("Upload market data (CSV) to forecast using a pre-trained LSTM model.")

# --- Sidebar UI ---
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
output_options = ['Adj Close', 'Close', 'Open', 'High', 'Low', 'Volume']

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

df = pd.read_csv('Market_cleaned_NYA')
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

    # --- Make predictions ---
    predictions = model.predict(X)

# --- Visualize predictions ---
st.subheader("📊 Predicted vs Actual")
actual = scaled_data[sequence_length:, 3]  # Actual 'Adj Close' column after seq offset
pred = predictions.flatten()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(actual, label="Actual (scaled)", color='blue')
ax.plot(pred, label="Predicted", color='orange')
ax.legend()
st.pyplot(fig)
