def calculate_rsi(data, periods=14):
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data (pd.Series): A pandas Series containing the closing prices.
        periods (int): The number of periods to use for the RSI calculation (default is 14).

    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    # Calculate daily price changes
    price_diff = data.diff()

    # Separate gains and losses
    gain = price_diff.clip(lower=0)
    loss = -1 * price_diff.clip(upper=0)

    # Calculate average gains and losses using a rolling mean
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()

    # Calculate Relative Strength (RS)
    # Handle division by zero for cases where avg_loss is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rs[np.isinf(rs)] = np.nan # Replace inf with NaN for consistency

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi



# Imports
from tensorflow import keras
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

import yfinance as yf
from yahoofinancials import YahooFinancials

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#yf.enable_debug_mode()

#data = yf.download('AAPL', start='2013-01-01', end='2025-09-01')
#close_prices = stock_data['Close']

#data = data.reset_index()
#data

#data = data.loc[:,['Date','Close']]
#print(data.columns)
data = pd.read_csv("MicrosoftStock.csv")

st.write(data.head())
st.write(data.info())
st.write(data.describe())
#close_prices = data['Close'] # This line caused an error with yfinance data with MultiIndex

# Calculate RSI
rsi_values = calculate_rsi(data['close']) # Pass the 'Close' column to the function
data['rsi']=rsi_values

data=data.fillna(0)


print(data.head(100))
print(data.info())
print(data.describe())

# Convert the Data into Date time then create a date filter



# Drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()

#data['Date'] = pd.to_datetime(data['Date'])
#prediction = data.loc[
#    (data['Date'] > datetime(2024,1,1)) &
#    (data['Date'] < datetime(2025,9,1))
#]

prediction=data

plt.figure(figsize=(12,6))
plt.plot(data['date'], data['rsi'],color="blue")
plt.xlabel("date")
plt.ylabel("rsi")
plt.title("Price over time")


# Prepare for the LSTM Model (Sequential)
#stock_close = data.filter(["rsi"]) # Filter creates a DataFrame with a MultiIndex if the original had one
stock_close = data['rsi'] # Select the 'rsi' column as a Series
dataset = stock_close.values.reshape(-1, 1) #convert to numpy array and reshape


training_data_len = int(np.ceil(len(dataset) * 0.95))



# Preprocessing Stages
#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(dataset)

#training_data = scaled_data[:training_data_len] #95% of all out data
training_data = dataset[:training_data_len] #95% of all out data


X_train, y_train = [], []


# Create a sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Build the Model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer (Dense)
#model.add(keras.layers.Dense(128, activation="tanh"))
model.add(keras.layers.Dense(128, activation="relu"))


# 4th Layer (Dropout)
model.add(keras.layers.Dropout(0.5))

# Final Output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])


training = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prep the test data
#test_data = scaled_data[training_data_len - 60:]

test_data = dataset[training_data_len - 60:]

X_test, y_test = [], dataset[training_data_len:]


for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

# Make a Prediction
predictions = model.predict(X_test)
#predictions= dataset.inverse_transform(predictions)
#predictions = scaler.inverse_transform(predictions)


# Plotting data
train = data[:training_data_len]
test =  data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['date'], train['rsi'], label="Train (Actual)", color='blue')
plt.plot(test['date'], test['rsi'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
#import plotly.exprss as px
#fig = px.scatter(

st.line_chart(train['date'], train['rsi'])
#st.line_chart(test['date'], test['rsi'], label="Test (Actual)", color='orange')
#st.line_chart(test['date'], test['Predictions'], label="Predictions", color='red')

plt.title("Our Stock Predictions based on RSI")
plt.xlabel("Date")
plt.ylabel("RSI Price")
plt.legend()
plt.show()

fig, ax = plt.subplots()
#ax.plot(x, y)
#ax.set_title("Sine Wave")

st.pyplot(fig)
