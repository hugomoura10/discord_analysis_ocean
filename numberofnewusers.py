import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')

df['Content'].fillna('N/A', inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M %p')
new_users_df = df[df['Content'].str.contains("Joined the server.")]
new_users_df['MonthYear'] = new_users_df['Date'].dt.to_period("M")

new_users_df['MonthYear'] = new_users_df['MonthYear'].dt.to_timestamp()

new_users_count_per_month = new_users_df.groupby('MonthYear').size().reset_index(name='NewUsersCount')

new_users_count_per_month.to_csv('new_users_count_per_month.csv', index=False)

print(new_users_count_per_month)

total_new_users_count = new_users_count_per_month['NewUsersCount'].sum()
print("Total number of new users:", total_new_users_count)

plt.figure(figsize=(12, 6))
plt.bar(new_users_count_per_month['MonthYear'].astype(str), new_users_count_per_month['NewUsersCount'], color='skyblue')
plt.title('Number of New Users per Month')
plt.xlabel('Month-Year')
plt.ylabel('New Users Count')
plt.xticks(rotation='vertical')
plt.show()

