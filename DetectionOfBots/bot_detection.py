import pandas as pd
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

twitter_df = pd.read_csv('DetectionOfBots/bot_detection_data_training.csv')  # Replace with training file path

twitter_features = twitter_df[['Tweet', 'Bot Label']]# Relevant columns for training
twitter_features.loc[:, 'Tweet'] = twitter_features['Tweet'].str.lower()


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(twitter_features['Tweet'])

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, twitter_features['Bot Label'])

discord_df = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')  # Replace with trained file path

discord_df['Content'].fillna('', inplace=True)
discord_df['Date'] = pd.to_datetime(discord_df['Date'], format='%m/%d/%Y %I:%M %p')
discord_df.loc[:, 'Content'] = discord_df['Content'].str.lower()
discord_vectorized = vectorizer.transform(discord_df['Content'])

# Predict 
discord_df['IsBot'] = naive_bayes_classifier.predict(discord_vectorized)

bot_threshold = 0.5
discord_df['IsBot'] = (naive_bayes_classifier.predict_proba(discord_vectorized)[:, 1] > bot_threshold).astype(int)

unique_bots = discord_df.loc[discord_df['IsBot'] == 1, 'Author'].unique()
print("Users Detected as Bots:")
print(pd.DataFrame({'Author': unique_bots, 'IsBot': 1}))

unique_bots_df = pd.DataFrame({'Author': unique_bots, 'IsBot': 1})
unique_bots_df.to_csv('DetectionOfBots/bots_detection_results.csv', index=False)
