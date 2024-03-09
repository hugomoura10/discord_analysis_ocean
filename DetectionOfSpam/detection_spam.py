import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the spam training data
spam_data = pd.read_csv('DetectionOfSpam/spam_ham.csv', encoding='latin-1')
# Use only the relevant columns
spam_data = spam_data[['v1', 'v2']]
# Rename columns for clarity
spam_data.columns = ['Label', 'Content']

# Map 'ham' to 0 (not spam) and 'spam' to 1 (spam)
spam_data['Label'] = spam_data['Label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spam_data['Content'], spam_data['Label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, y_train)

# Load the actual CSV file with the messages
actual_data = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')

# Handle missing values in the 'Content' column
actual_data['Content'].fillna('', inplace=True)

# Vectorize the actual data
actual_data_vectorized = vectorizer.transform(actual_data['Content'])

# Predict if the content is spam or not
actual_data['IsSpam'] = naive_bayes_classifier.predict(actual_data_vectorized)

# Save the results to a new CSV file
actual_data.to_csv('DetectionOfSpam/spam_detection_results_with_authors.csv', index=False)

# Filter out messages classified as spam
spam_messages = actual_data[actual_data['IsSpam'] == 1]

# Save the spam messages to a new CSV file
spam_messages.to_csv('DetectionOfSpam/spam_messages.csv', index=False)

# Display the first few rows of the spam messages
print('\nSpam Messages:')
print(spam_messages[['Content', 'IsSpam']].head())

# Extract the most common words in spam messages
def get_most_common_words(data, top_n=40):
    vectorizer_spam = CountVectorizer(stop_words='english')
    X_spam = vectorizer_spam.fit_transform(data)
    word_freq_spam = pd.DataFrame(X_spam.toarray(), columns=vectorizer_spam.get_feature_names_out())

    # Exclude words containing "ocean"
    ocean_related_words = [word for word in word_freq_spam.columns if 'ocean' in word.lower()]
    word_freq_spam = word_freq_spam.drop(columns=ocean_related_words, errors='ignore')

    most_common_words_spam = word_freq_spam.sum().sort_values(ascending=False).head(top_n)

    # Plot Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(most_common_words_spam.to_dict())
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud of Most Common Words in Spam Messages')
    plt.show()

    return most_common_words_spam

# Get the 20 most common words in spam messages and plot the Word Cloud
most_common_words_spam = get_most_common_words(spam_messages['Content'], top_n=40)

# Plot the bar chart of the most common words
plt.figure(figsize=(10, 6))
most_common_words_spam.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Top 20 Most Common Words in Spam Messages')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()
