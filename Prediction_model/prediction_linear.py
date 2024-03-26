import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Monthly_Message_Count.csv')
df['Date'] = pd.to_datetime(df['Date'])

price_data = pd.read_csv('Datasets/OCEAN-USD-4.csv')
price_data['Date'] = pd.to_datetime(price_data['Date'])

merged_data = pd.merge(df, price_data, on='Date', how='left')
merged_data.dropna(subset=['Close'], inplace=True)
X = merged_data[['Close']]
y = merged_data['MessageCount']
model = LinearRegression()

model.fit(X, y)
future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')  # Adjust the date 
future_df = pd.DataFrame({'Date': future_dates})

future_df = pd.merge(future_df, price_data, on='Date', how='left')

future_df['Close'].fillna(future_df['Close'].mean(), inplace=True)
future_df['Predicted_MessageCount'] = model.predict(future_df[['Close']])
mae = mean_absolute_error(y, model.predict(X))
print(f'Mean Absolute Error on Historical Data: {mae}')

plt.plot(merged_data['Date'], merged_data['MessageCount'], label='Historical Data', marker='o', linestyle='-', color='b')
plt.plot(future_df['Date'], future_df['Predicted_MessageCount'], label='Predicted', marker='o', linestyle='-', color='r')
plt.title('Predicted Message Count for Future Dates')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.legend()
plt.show()
