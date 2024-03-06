import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator

# Load the data from the CSV file
file_path = 'Datasets/OCEAN-USD-4.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

plt.figure(figsize=(15, 10))
plt.plot(df['Date'], df['Close'], marker='o', linestyle='-', color='b', linewidth=1, markersize=1)

locator = MonthLocator(interval=2)
plt.gca().xaxis.set_major_locator(locator)

# Plotting with a thinner line and smaller markers
plt.title('Ocean Protocol USD Price Over Time')
plt.xlabel('Date')
plt.ylabel('USD Price')
plt.xticks(rotation='vertical')
plt.show()
