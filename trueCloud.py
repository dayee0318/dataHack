import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from textblob import TextBlob
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the data into a pandas dataframe
df = pd.read_csv('DataSet_Misinfo_TRUE.csv')

# Download stopwords and punkt tokenizer from NLTK
nltk.download('stopwords')  # Common words without meaning
nltk.download('averaged_perceptron_tagger') # part of speech tag
nltk.download('punkt') # punctuation tokenizer

# Define stopwords and punctuation to remove from the text
stop_words = set(stopwords.words('english'))
punct = set(punctuation)

# Function to preprocess the text
def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token not in stop_words and token not in punct]
        # Join the tokens back into a single string
        return ' '.join(tokens)
    else:
        # Return an empty string for missing values
        return ''

# Apply the preprocessing function to the text column
df['text'] = df['text'].apply(preprocess_text)

# Drop rows with missing values
df = df.dropna()

# Sentiment analysis
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Named entity recognition
nltk.download('maxent_ne_chunker')
nltk.download('words')

def get_entities(text):
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))
    return entities

df['entities'] = df['text'].apply(get_entities)

# Generate word cloud
text = ' '.join(df['text'])
wordcloud = WordCloud(width=800, height=800, background_color='white')
wordcloud.generate(text)

# Get the 100 most frequent words and their frequencies
word_freq = sorted(wordcloud.process_text(text).items(), key=lambda x: x[1], reverse=True)[:100]

# Create a dictionary to store the frequency total and number of article for each word
freq_dict = {}
article_dict = {}

for word, freq in word_freq:
    freq_dict[word] = freq
    article_dict[word] = len(df[df['text'].str.contains(word)])

# Convert the dictionaries to pandas dataframes
df_freq = pd.DataFrame(list(freq_dict.items()), columns=['word', 'freq'])
df_article = pd.DataFrame(list(article_dict.items()), columns=['word', 'article_count'])

# Merge the dataframes on the 'word' column
df_result = pd.merge(df_freq, df_article, on='word')

# Calculate the rank based on the word frequency
df_result['rank'] = df_result['freq'].rank(ascending=False)

# Create a CSV file with the data
os.makedirs('result', exist_ok=True)
most_freq_file = os.path.join('result', 'trueData_Most.csv')

df_result.to_csv(most_freq_file, index=False)

# Print the results
print('100 most frequent words in the text:\n')
print(df_result[['rank', 'word', 'freq', 'article_count']].head(100))

# Plot the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Print the results
print('100 most frequent words in the text:\n')
print(df_result[['rank', 'word', 'freq', 'article_count']].head(100))
