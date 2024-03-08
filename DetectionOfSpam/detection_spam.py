import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the spam training data
spam_data = pd.read_csv('Datasets/spam.csv', encoding='latin-1')
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

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2%}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Load the actual CSV file with the messages
actual_data = pd.read_csv('Datasets/Ocean Discord Data Challenge Dataset.csv')

# Handle missing values in the 'Content' column
actual_data['Content'].fillna('', inplace=True)

# Vectorize the actual data
actual_data_vectorized = vectorizer.transform(actual_data['Content'])

# Predict if the content is spam or not
actual_data['IsSpam'] = naive_bayes_classifier.predict(actual_data_vectorized)

# Save the results to a new CSV file
actual_data.to_csv('spam_detection_results.csv', index=False)

# Display the first few rows of the results
print('\nSpam Detection Results:')
print(actual_data[['Content', 'IsSpam']].head())

# Filter out messages classified as spam
spam_messages = actual_data[actual_data['IsSpam'] == 1]

# Save the spam messages to a new CSV file
spam_messages.to_csv('spam_messages.csv', index=False)

# Display the first few rows of the spam messages
print('\nSpam Messages:')
print(spam_messages[['Content', 'IsSpam']].head())
