import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Load CSV file into a Pandas DataFrame with 'Channel' as header
df = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M %p')

# Check for missing values and fill with 'N/A'
df['Content'].fillna('N/A', inplace=True)

# Create a SentimentIntensityAnalyzer object
sid = SentimentIntensityAnalyzer()

# Create 'Datasets' folder if it doesn't exist
output_folder = 'Channels'
os.makedirs(output_folder, exist_ok=True)

# Iterate through unique channels and save filtered data to CSV files
for channel in df['Channel'].unique():
    # Subset the data for the current channel
    channel_data = df[df['Channel'] == channel]

    # Apply sentiment analysis to 'Content' column
    channel_data['sentiment'] = channel_data['Content'].apply(lambda text: sid.polarity_scores(text)['compound'])

    # Resample data to monthly frequency and calculate average sentiment
    monthly_sentiment = channel_data.resample('MS', on='Date')['sentiment'].agg(['mean', 'count'])

    # Save the filtered data to a new CSV file in the 'Datasets' folder without 'Date' as the index
    output_file = os.path.join(output_folder, f'{channel}_data.csv')
    channel_data.to_csv(output_file, index=False)
