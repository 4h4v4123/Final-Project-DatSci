import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import pickle
from prophet.diagnostics import cross_validation, performance_metrics
from io import BytesIO

# App title
st.title("📈 Time Series Forecasting with Prophet")

def load_model():
    try:
        with open('prophet_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Upload CSV file
uploaded_file = st.file_uploader("Upload your time series CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded data:")
    st.dataframe(df.head())

    # Check column names
    if "ds" in df.columns and "y" in df.columns:
        # Filter date range
        min_date, max_date = df['ds'].min(), df['ds'].max()
        start_date = st.date_input("Start date", pd.to_datetime(min_date))
        end_date = st.date_input("End date", pd.to_datetime(max_date))

        filtered_df = df[(pd.to_datetime(df['ds']) >= pd.to_datetime(start_date)) &
                         (pd.to_datetime(df['ds']) <= pd.to_datetime(end_date))]

        st.write("Filtered data:")
        st.dataframe(filtered_df)

        # Prophet parameters
        st.sidebar.subheader("Prophet Parameters")
        seasonality_mode = st.sidebar.selectbox("Seasonality Mode", options=["additive", "multiplicative"])

        # Holiday effects
        add_holidays = st.sidebar.checkbox("Include Holidays", value=False)
        holidays = None
        if add_holidays:
            country = st.sidebar.selectbox("Select Country for Holidays", ["US", "ID", "UK", "IN", "JP"])
            from prophet.make_holidays import make_holidays_df
            holidays = make_holidays_df(year_list=list(range(2000, 2031)), country=country)

        # Prophet modeling
        model = Prophet(seasonality_mode=seasonality_mode, holidays=holidays)
        model.fit(filtered_df)

        # Forecasting future
        periods_input = st.number_input("How many future periods would you like to forecast?", min_value=1, max_value=365, value=30)
        future = model.make_future_dataframe(periods=periods_input)
        forecast = model.predict(future)

        # Plotting the forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        st.plotly_chart(fig)

        st.subheader("Forecast Data")
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input)
        st.dataframe(forecast_data)

        # Download button for forecast
        csv = forecast_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast Data as CSV",
            data=csv,
            file_name='forecast.csv',
            mime='text/csv'
        )

        # Show forecast components
        st.subheader("Forecast Components")
        components_fig = model.plot_components(forecast)
        st.write(components_fig)

        # Cross-validation and performance
        st.subheader("Model Performance (Cross-Validation)")
        horizon = st.selectbox("Select horizon for cross-validation", options=["7 days", "30 days", "90 days"], index=1)
        try:
            df_cv = cross_validation(model, initial='180 days', period='30 days', horizon=horizon)
            df_p = performance_metrics(df_cv)
            st.write("Performance Metrics:")
            st.dataframe(df_p)
        except Exception as e:
            st.warning(f"Cross-validation could not be performed: {e}")

    else:
        st.error("CSV must contain 'ds' and 'y' columns.")