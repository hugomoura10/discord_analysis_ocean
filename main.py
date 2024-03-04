import pandas as pd

# Load CSV file into a Pandas DataFrame
df = pd.read_csv('Ocean Discord Data Challenge Dataset.csv')

#print(df.info())
#print(df.describe())
#print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

# Plot time series data
df.set_index('Date')['Content'].resample('D').count().plot(title='Daily Message Count')
