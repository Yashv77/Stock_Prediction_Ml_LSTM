import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Function to load and preprocess data (cached)
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    closing_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    closing_prices_scaled = scaler.fit_transform(closing_prices)
    return closing_prices_scaled, scaler, df.index

# Function to create sequences for LSTM (cached)
@st.cache_data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Function to build LSTM model (cached)
@st.cache_resource
def build_model(sequence_length):
    model = keras.Sequential([
        layers.LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)),
        layers.RepeatVector(sequence_length),
        layers.LSTM(units=50, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train the model and make predictions (cached)
@st.cache_data
def train_and_predict(stock_name, file_path):
    data_scaled, scaler, dates = load_and_preprocess_data(file_path)
    sequence_length = 10
    X = create_sequences(data_scaled, sequence_length)

    # Split into train and test
    train_size = int(len(X) * 0.80)
    train, test = X[:train_size], X[train_size:]

    # Train the model with progress bar
    model = build_model(sequence_length)
    epochs = 50  # Number of epochs
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        history = model.fit(train, train, epochs=1, batch_size=32, validation_split=0.1, verbose=0)
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoch {epoch + 1}/{epochs} completed')

    # Clear the progress bar and status text after training
    progress_bar.empty()
    status_text.empty()

    # Make predictions
    predictions = model.predict(test)
    predictions_inverse = scaler.inverse_transform(predictions[:, -1, 0].reshape(-1, 1))
    actual_inverse = scaler.inverse_transform(test[:, -1, 0].reshape(-1, 1))

    return predictions_inverse, actual_inverse, history.history['loss'], history.history['val_loss'], dates[-len(actual_inverse):]

# File paths for each stock
file_paths = {
    'Apple': 'AAPL.csv',
    'Meta': 'META.csv',
    'Google': 'GOOG.csv'
}

# Streamlit App
st.title("Stock Price Prediction using LSTM")

st.header("Introduction")
st.write("""
This project aims to predict the future stock prices of Apple, Meta, and Google using a Long Short-Term Memory (LSTM) neural network. 
LSTMs are a type of recurrent neural network (RNN) that are well-suited for time series data and sequential information.
""")

st.header("Problem Statement")
st.write("""
Predicting stock prices is a challenging task due to the complex, non-linear, and volatile nature of stock market data. 
This project attempts to use deep learning to provide insights and predictions based on historical data.
""")

# Stock selection
st.header("Select a Stock to View Results")
stock_name = st.selectbox("Choose a stock", options=["Apple", "Meta", "Google"])

# Display results
if stock_name:
    st.subheader(f"Results for {stock_name}")
    file_path = file_paths[stock_name]
    predictions, actual, train_loss, val_loss, dates = train_and_predict(stock_name, file_path)

    # Plot Actual vs Predicted
    st.write("### Actual vs Predicted Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, actual, label='Actual Prices')
    ax.plot(dates, predictions, label='Predicted Prices', linestyle='--')
    ax.set_title(f'Actual vs Predicted Stock Prices for {stock_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

    # Plot Training vs Validation Loss
    st.write("### Training vs Validation Loss")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_title(f'Training vs Validation Loss for {stock_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
