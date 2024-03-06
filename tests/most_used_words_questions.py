import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load the CSV file with the messages
file_path = 'Channels/Ocean Protocol - GET STARTED - ask-the-ai [1082698926865522808]_data.csv'
df = pd.read_csv(file_path)

# Handle missing values in the 'Content' column
df['Content'].fillna('', inplace=True)

# Function to extract the most common words
def get_most_common_words(data, top_n=20):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    most_common_words = word_freq.sum().sort_values(ascending=False).head(top_n)
    return most_common_words

# Define a list of common question words
question_words = ['what', 'where', 'when', 'why', 'who', 'which', 'how']

# Filter out rows with questions
questions_df = df[df['Content'].apply(lambda x: any(word in x.lower() for word in question_words) or '?' in x)]

# Get the 10 most common words in questions
most_common_words_questions = get_most_common_words(questions_df['Content'], top_n=20)
print(most_common_words_questions)
# Plot the results
plt.figure(figsize=(10, 6))
most_common_words_questions.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 20 Most Common Words in Questions')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

