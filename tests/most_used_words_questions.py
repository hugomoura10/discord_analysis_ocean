import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

file_path = 'Channels/Ocean Protocol - GET STARTED - ask-the-ai [1082698926865522808]_data.csv'
df = pd.read_csv(file_path)

df['Content'].fillna('', inplace=True)

def get_most_common_words(data, top_n=20):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    most_common_words = word_freq.sum().sort_values(ascending=False).head(top_n)
    return most_common_words

question_words = ['what', 'where', 'when', 'why', 'who', 'which', 'how'] #common words

questions_df = df[df['Content'].apply(lambda x: any(word in x.lower() for word in question_words) or '?' in x)]

most_common_words_questions = get_most_common_words(questions_df['Content'], top_n=20)
print(most_common_words_questions)
plt.figure(figsize=(10, 6))
most_common_words_questions.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 20 Most Common Words in Questions')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

