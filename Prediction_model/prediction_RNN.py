import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load your message count dataset
df = pd.read_csv('Datasets/Monthly_Message_Count.csv')

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Load the additional dataset for the coin price
price_data = pd.read_csv('Datasets/OCEAN-USD-4.csv')

# Ensure the 'Date' column is in datetime format
price_data['Date'] = pd.to_datetime(price_data['Date'])

# Merge datasets on 'Date'
merged_data = pd.merge(df, price_data, on='Date', how='left')

# Drop missing values
merged_data.dropna(subset=['Close'], inplace=True)

# Prepare features and target variable
X = merged_data[['Close']]
y = merged_data['MessageCount']

# Train an ARIMA model
model = ARIMA(y, order=(3,1,2))  # Adjust order based on your data characteristics
result = model.fit()

# Create a dataframe for future dates
future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')  # Adjust the date range as needed
future_df = pd.DataFrame({'Date': future_dates})

# Merge the future dataframe with coin price data
future_df = pd.merge(future_df, price_data, on='Date', how='left')

# Forecast using the trained ARIMA model
forecast = result.get_forecast(steps=len(future_df), alpha=0.05)

# Calculate the Mean Absolute Error on historical data
mae_historical = mean_absolute_error(y, result.fittedvalues)

# Print the Mean Absolute Error on historical data
print(f"Mean Absolute Error on Historical Data: {mae_historical}")

# Plot the predicted message count for future dates
plt.plot(merged_data['Date'], merged_data['MessageCount'], label='Historical Data', marker='o', linestyle='-', color='b')
plt.plot(future_df['Date'], forecast.predicted_mean, label='Predicted', marker='o', linestyle='-', color='r')
plt.title('Predicted Message Count for Future Dates using ARIMA')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.legend()
plt.show()

# Print the mean forecast values
print("Mean Forecast:")
print(forecast.predicted_mean)
