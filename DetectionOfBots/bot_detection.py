import pandas as pd
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the Twitter training set
twitter_df = pd.read_csv('DetectionOfBots/bot_detection_data_training.csv')  # Replace with the actual file path

# Select relevant columns for training
twitter_features = twitter_df[['Tweet', 'Bot Label']]

# Convert 'Tweet' column to lowercase using loc accessor
twitter_features.loc[:, 'Tweet'] = twitter_features['Tweet'].str.lower()

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(twitter_features['Tweet'])

# Train a Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, twitter_features['Bot Label'])

# Load the Ocean Discord dataset
discord_df = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')  # Replace with the actual file path

# Handle missing values in the 'Content' column
discord_df['Content'].fillna('', inplace=True)

# Convert 'Date' column to datetime format
discord_df['Date'] = pd.to_datetime(discord_df['Date'], format='%m/%d/%Y %I:%M %p')

# Convert 'Content' column to lowercase using loc accessor
discord_df.loc[:, 'Content'] = discord_df['Content'].str.lower()

# Vectorize the actual data
discord_vectorized = vectorizer.transform(discord_df['Content'])

# Predict if the content is from a bot using the trained classifier
discord_df['IsBot'] = naive_bayes_classifier.predict(discord_vectorized)

# Set a threshold for bot detection (e.g., 0.5 for the predicted probability)
bot_threshold = 0.5
discord_df['IsBot'] = (naive_bayes_classifier.predict_proba(discord_vectorized)[:, 1] > bot_threshold).astype(int)

# Print users detected as bots
unique_bots = discord_df.loc[discord_df['IsBot'] == 1, 'Author'].unique()
print("Users Detected as Bots:")
print(pd.DataFrame({'Author': unique_bots, 'IsBot': 1}))

unique_bots_df = pd.DataFrame({'Author': unique_bots, 'IsBot': 1})
unique_bots_df.to_csv('DetectionOfBots/bots_detection_results.csv', index=False)
