import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import os

df = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M %p')

df['Content'].fillna('N/A', inplace=True)

sid = SentimentIntensityAnalyzer()

output_folder = 'Channels'
os.makedirs(output_folder, exist_ok=True)

for channel in df['Channel'].unique():
    channel_data = df[df['Channel'] == channel]

    channel_data['sentiment'] = channel_data['Content'].apply(lambda text: sid.polarity_scores(text)['compound'])

    monthly_sentiment = channel_data.resample('MS', on='Date')['sentiment'].agg(['mean', 'count'])

    output_file = os.path.join(output_folder, f'{channel}_data.csv')
    channel_data.to_csv(output_file, index=False)
