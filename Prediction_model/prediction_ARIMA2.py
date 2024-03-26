import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('Datasets/Monthly_Message_Count.csv')
df['Date'] = pd.to_datetime(df['Date'])

X = df[['Date']]
y = df['MessageCount']

# Train an ARIMA model
model = ARIMA(y, order=(3, 1, 2))
result = model.fit()
future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')  # Adjust the date range as needed
future_df = pd.DataFrame({'Date': future_dates})

# Forecast
forecast = result.get_forecast(steps=len(future_df), alpha=0.05)
mae_historical = mean_absolute_error(y, result.fittedvalues)
print(f"Mean Absolute Error on Historical Data: {mae_historical}")
plt.plot(df['Date'], y, label='Historical Data', marker='o', linestyle='-', color='b')
plt.plot(future_df['Date'], forecast.predicted_mean, label='Predicted', marker='o', linestyle='-', color='r')
plt.title('Predicted Message Count for Future Dates using ARIMA')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.legend()
plt.show()

print("Mean Forecast:")
print(forecast.predicted_mean)
