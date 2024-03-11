#(c) 2023 SMDS-Studio
from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from matplotlib.dates import YearLocator

sent = pd.read_csv('Datasets/Monthly_Sentiment.csv')
sent['Date'] = pd.to_datetime(sent['Date'])
sent = sent[['Date', 'mean']]
sent.columns = ['ds', 'y']

fig, ax = plt.subplots()
ax.plot(sent['ds'], sent['y'], label='actual', color='g')

# Set the locator for the x-axis to show years
ax.xaxis.set_major_locator(YearLocator())

# Format the date labels to display only the years
plt.xticks(rotation=45, ha='right')

plt.legend()
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.show()

model = NeuralProphet()
model.fit(sent)

future = model.make_future_dataframe(sent, periods = 55)

forecast = model.predict(future)
actual_prediction = model.predict(sent)

plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label = "prediction_Actual", c = 'r')
plt.plot(forecast['ds'], forecast['yhat1'], label = 'future_prediction', c = 'b')
plt.plot(sent['ds'], sent['y'], label = 'actual', c = 'g')
plt.legend()
plt.title('Prediction')
plt.show()

model.plot_components(forecast)
mae_historical = mean_absolute_error(actual_prediction['y'], actual_prediction['yhat1'])
print(f"Mean Absolute Error on Historical Data: {mae_historical}")

# Calculate MAE for future predictions
print("Length of Sentiment Data:", len(sent['y']))
print("Length of Forecast Data:", len(forecast['yhat1']))

# Ensure that the lengths match before calculating MAE
if len(sent['y']) == len(forecast['yhat1']):
    mae_future = mean_absolute_error(sent['y'], forecast['yhat1'][:len(sent['y'])])
    print(f"Mean Absolute Error on Future Data: {mae_future}")
else:
    print("Lengths of sentiment and forecast data are inconsistent.")