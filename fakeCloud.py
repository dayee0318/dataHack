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
df = pd.read_csv('DataSet_Misinfo_FAKE.csv')

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

# Get the 50 most frequent words and their frequencies
word_freq = sorted(wordcloud.process_text(text).items(), key=lambda x: x[1], reverse=True)[:50]

# Convert the list of tuples to a pandas dataframe
df_freq = pd.DataFrame(word_freq, columns=['word', 'freq'])

# Print the results
print('50 most frequent words in the text:\n')
print(df_freq)

# Plot the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
