from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from matplotlib.dates import YearLocator


data = pd.read_csv('Datasets/Monthly_Message_Count.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data[['Date', 'MessageCount']]
data.columns = ['ds', 'y']

fig, ax = plt.subplots()
ax.plot(data['ds'], data['y'], label='actual', color='g')

# Set the locator for the x-axis to show years
ax.xaxis.set_major_locator(YearLocator())

# Format the date labels to display only the years
plt.xticks(rotation=45, ha='right')

plt.legend()
plt.title('Monthly Message Count Over Time')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.show()

model = NeuralProphet()
model.fit(data)

future = model.make_future_dataframe(data, periods = 55)

forecast = model.predict(future)
actual_prediction = model.predict(data)

plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label = "prediction_Actual", c = 'r')
plt.plot(forecast['ds'], forecast['yhat1'], label = 'future_prediction', c = 'b')
plt.plot(data['ds'], data['y'], label = 'actual', c = 'g')
plt.legend()
plt.title('Prediction')
plt.show()

model.plot_components(forecast)
mae_historical = mean_absolute_error(actual_prediction['y'], actual_prediction['yhat1'])
print(f"Mean Absolute Error on Historical Data: {mae_historical}")

# Check the lengths of actual and forecast for future predictions
print("Length of Actual Data:", len(data['y']))
print("Length of Forecast Data:", len(forecast['yhat1']))

# Calculate MAE for future predictions if lengths match
if len(data['y']) == len(forecast['yhat1']):
    mae_future = mean_absolute_error(data['y'], forecast['yhat1'])
    print(f"Mean Absolute Error on Future Data: {mae_future}")
else:
    print("Lengths of actual and forecast data are inconsistent.")
