import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from textblob import TextBlob
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the data into pandas dataframes
true_df = pd.read_csv('DataSet_Misinfo_TRUE.csv')
fake_df = pd.read_csv('DataSet_Misinfo_FAKE.csv')

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

# Apply the preprocessing function to the text column of both dataframes
true_df['text'] = true_df['text'].apply(preprocess_text)
fake_df['text'] = fake_df['text'].apply(preprocess_text)

# Drop rows with missing values
true_df = true_df.dropna()
fake_df = fake_df.dropna()

# Sentiment analysis
true_df['sentiment'] = true_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
fake_df['sentiment'] = fake_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

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


true_df['entities'] = true_df['text'].apply(get_entities)
fake_df['entities'] = fake_df['text'].apply(get_entities)

# Generate word clouds
true_text = ' '.join(true_df['text'])
fake_text = ' '.join(fake_df['text'])

true_wordcloud = WordCloud(width=800, height=800, background_color='white').generate(true_text)
fake_wordcloud = WordCloud(width=800, height=800, background_color='white').generate(fake_text)

# Plot the word clouds
plt.imshow(true_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# As the .csv files are too big, I split the process into two so that one .py can handle just one .csv file
