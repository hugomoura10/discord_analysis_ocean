import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the entire dataset
model.fit(X, y)

# Create a dataframe for future dates
future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')  # Adjust the date range as needed
future_df = pd.DataFrame({'Date': future_dates})

# Merge the future dataframe with coin price data
future_df = pd.merge(future_df, price_data, on='Date', how='left')

# Fill NaN values in 'Close' column with the mean
future_df['Close'].fillna(future_df['Close'].mean(), inplace=True)

# Predict message count for future dates based on coin price
future_df['Predicted_MessageCount'] = model.predict(future_df[['Close']])

# Print Mean Absolute Error (MAE)
mae = mean_absolute_error(y, model.predict(X))
print(f'Mean Absolute Error on Historical Data: {mae}')

# Plot the predicted message count for future dates
plt.plot(merged_data['Date'], merged_data['MessageCount'], label='Historical Data', marker='o', linestyle='-', color='b')
plt.plot(future_df['Date'], future_df['Predicted_MessageCount'], label='Predicted', marker='o', linestyle='-', color='r')
plt.title('Predicted Message Count for Future Dates')
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.legend()
plt.show()
