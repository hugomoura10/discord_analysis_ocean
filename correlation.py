import pandas as pd

#main dataset
message_count_df = pd.read_csv('Datasets/Monthly_Message_Count.csv')
message_count_df['Date'] = pd.to_datetime(message_count_df['Date'], format='%Y-%m-%d')

#sentiment dataset
sentiment_df = pd.read_csv('Datasets/Monthly_Sentiment.csv')
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'], format='%Y-%m-%d')

#price dataset
price_df = pd.read_csv('Datasets/Ocean Protocol Historical Data.csv')
price_df['Date'] = pd.to_datetime(price_df['Date'], format='%m/%d/%Y')  # Adjust the format accordingly

new_users_df = pd.read_csv('Datasets/Monthly_New_Users.csv')
new_users_df['Date'] = pd.to_datetime(new_users_df['Date'], format='%Y-%m-%d')  # Adjust the format accordingly

merged_df = pd.merge(message_count_df, sentiment_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, price_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, new_users_df, on='Date', how='inner')


# Calculate correlation between diff. variables
correlation_price_message = merged_df['MessageCount'].corr(merged_df['Price'])
correlation_price_sentiment = merged_df['mean'].corr(merged_df['Price'])
correlation_price_new_users = merged_df['NewUsersCount'].corr(merged_df['Price'])

print(f'Correlation between message count and price: {correlation_price_message}')
print(f'Correlation between sentiment and price: {correlation_price_sentiment}')
print(f'Correlation between new users and price: {correlation_price_new_users}')