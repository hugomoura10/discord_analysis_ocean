import pandas as pd

# Load your daily message count dataset
df = pd.read_csv('Datasets/Daily_Message_Count.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Resample data to monthly frequency and sum the message counts
monthly_message_count = df.resample('MS').sum()

# Reset the index to make 'Date' a regular column again
monthly_message_count.reset_index(inplace=True)

# Print the result
print(monthly_message_count)

# Save the result to a new CSV file
monthly_message_count.to_csv('path_to_monthly_message_count.csv', index=False)