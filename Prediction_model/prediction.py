import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your dataset with Date and MessageCount columns
df = pd.read_csv('Datasets/Daily_Message_Count.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Ensure the dataset is sorted by date
df.sort_values(by='Date', inplace=True)

# Train-test split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['MessageCount'], label='Message Count')
plt.title('Message Count Over Time')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.legend()
plt.show()

# Fit ARIMA model
model = ARIMA(train['MessageCount'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Evaluate the model
rmse = sqrt(mean_squared_error(test['MessageCount'], predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['MessageCount'], label='Training Data')
plt.plot(test['Date'], test['MessageCount'], label='Actual Message Count')
plt.plot(test['Date'], predictions, label='Predicted Message Count')
plt.title('ARIMA Forecast for Message Count')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.legend()
plt.show()