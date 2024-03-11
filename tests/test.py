import pandas as pd

df = pd.read_csv('Datasets/Daily_Message_Count.csv')
df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

monthly_message_count = df.resample('MS').sum()

monthly_message_count.reset_index(inplace=True)

print(monthly_message_count)

