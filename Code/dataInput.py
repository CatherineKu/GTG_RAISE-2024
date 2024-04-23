#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:32:55 2024

@author: chuhanku
"""

#NLP 统计词语多少
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Dataset_10k (1).csv')

#print(df['translated_title'])

language_counts = df['language'].value_counts()
total_languages = language_counts.count()

print("Total number of different languages:", total_languages)
print("\nCounts for each language:")
print(language_counts)

tld_counts = df['top_level_domain'].value_counts()
total_tlds = tld_counts.count()

print("Total number of different top-level domains:", total_tlds)
print("\nCounts for each top-level domain:")
print(tld_counts)


tld_counts_df = pd.DataFrame({'Top Level Domain': tld_counts.index, 'Count': tld_counts.values})

# Output DataFrame to CSV file
tld_counts_df.to_csv('tld_counts.csv', index=False)

# Print DataFrame
print(tld_counts_df)


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


import nltk
#from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


# Download WordNet dataset
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the preprocess function
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatizer.lemmatize(token))
    return result

# Apply the preprocess function to each row in the "translated_title" column
df['preprocessed_title'] = df['translated_title'].apply(preprocess)

# Display the preprocessed titles
#print(df['preprocessed_title'])


# Define a function to perform text preprocessing
import re

def preprocess_text(text):
    # Remove special characters, non-alphabetic characters, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply text preprocessing to the 'translated_title' column
df['translated_title'] = df['translated_title'].apply(preprocess_text)

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the preprocessed text data into a bag-of-words representation
bow_matrix = vectorizer.fit_transform(df['translated_title'])

# Convert the bag-of-words matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Optionally, concatenate the bag-of-words DataFrame with the original DataFrame
# new_df = pd.concat([df, bow_df], axis=1)

# Print the bag-of-words DataFrame



# Save the bag-of-words DataFrame to a CSV file
#bow_df.to_csv('bag_of_words.csv', index=False)
# Save the bag-of-words DataFrame to a CSV file
# bow_df.to_csv('bag_of_words.csv', index=False)
'''
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk

if __name__ == '__main__':
    # Assuming 'texts' is your tokenized corpus
    dictionary = gensim.corpora.Dictionary(df['preprocessed_title'])
    corpus = [dictionary.doc2bow(title) for title in df['preprocessed_title']]
    
    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary)
    
    # Calculate coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['preprocessed_title'], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    # Calculate perplexity
    perplexity = lda_model.log_perplexity(corpus)
    
    print(coherence_lda)
    print(perplexity)
'''

import os
import random
import spacy
from spacy.util import minibatch, compounding
import pandas as pd
'''
df = pd.read_csv("Dataset_10k (1).csv")

df['translated_title']

#from langdetect import detect

#change all strings to be lower
df['translated_title']=df['translated_title'].str.lower()

#get rid of unwanted characters such as punctuation marks
df['translated_title']=df['translated_title'].str.replace('[^\w\s]','')

#removing numerals
df['translated_title']=df['translated_title'].str.replace('\d+','') 

df['preprocessed_title']=df['translated_title'].str.replace('\n',' ').str.replace('\r','')

'''


#LDA
if __name__ == '__main__':
    import gensim
    #from gensim.utils import simple_preprocess
    #from gensim.parsing.preprocessing import STOPWORDS
    #from nltk.stem import WordNetLemmatizer
    #import numpy as np
    import nltk
    nltk.download('wordnet')

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define the preprocess function
    def preprocess(text):
        result = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS:
                result.append(lemmatizer.lemmatize(token))
        return result

    # Apply preprocessing to each title in the 'translated_title' column
    df['preprocessed_title'] = df['translated_title'].apply(preprocess)

    # Create a dictionary from the preprocessed titles
    dictionary = gensim.corpora.Dictionary(df['preprocessed_title'])

    # Create a corpus (bag of words) from the preprocessed titles
    corpus = [dictionary.doc2bow(title) for title in df['preprocessed_title']]
    
    # Define the words to remove
    words_to_remove = {"ai", "artificial intelligence", "artificial", "gpt", "chatbot", "intelligence"}

# Create the corpus without the specified words
    corpus_without_words = [
    dictionary.doc2bow([token for token in title if token not in words_to_remove]) 
    for title in df['preprocessed_title']
    ]

    # Train the LDA model
    lda_model = gensim.models.LdaMulticore(corpus_without_words, num_topics=5, id2word=dictionary, passes=10, workers=2)

    # Print the topics
    #for idx, topic in lda_model.print_topics(-1):
        #print("Topic: {} Words: {}".format(idx, topic))
        
# Assign topics to each document
    topic_assignments = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus_without_words]

# Map topic IDs to their corresponding labels
    topic_map = {0: "chat gpt", 1: "huminity", 2: "Finance+Business", 3: "new", 4: "language model (tech)"}

# Create a new column "column_name" in the DataFrame df to store the topic assignments
    df['column_name'] = [topic_map[topic_id] for topic_id in topic_assignments]

# Print the DataFrame to verify the new column

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each title in the 'preprocessed_title' column
    sentiment_scores = df['preprocessed_title'].apply(lambda x: sid.polarity_scores(' '.join(x)))

# Extract compound sentiment scores
    compound_scores = sentiment_scores.apply(lambda x: x['compound'])

# Assign sentiment labels based on the compound scores
    sentiment_labels = compound_scores.apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Add sentiment labels to the DataFrame as a new column
    df['sentiment'] = sentiment_labels
    
    file_path = 'output.csv'
    
    
    # count the num of positive_no negative_no and neutral_no for all data --- percentiles
    # Count occurrences of each sentiment label
    sentiment_counts = df['sentiment'].value_counts()

# Calculate total count of sentiments
    total_sentiments = sentiment_counts.sum()

# Calculate percentage of each sentiment label
    sentiment_percentages = sentiment_counts / total_sentiments * 100

# Calculate percentiles

# Output results
    print("Sentiment Counts:")
    print(sentiment_counts)
    print("\nSentiment Percentages:")
    print(sentiment_percentages)
    
    
    
# Select only the columns 'translated_title', 'column_name', and 'sentiment'
    selected_columns = ['translated_title', 'column_name']
    selected_df = df[selected_columns]

# Output the selected DataFrame to a CSV file
    selected_df.to_csv(file_path, index=False)
    

# Print the DataFrame with the sentiment labels
    sentiment_counts = df.groupby('column_name')['sentiment'].value_counts().unstack().fillna(0)
# Group the DataFrame by the 'column_name' column and calculate the sentiment counts for each group
    print(sentiment_counts)
    '''
# Rename the index and columns for clarity
    sentiment_counts.index.name = 'category'
    sentiment_counts.columns = ['positive_no', 'negative_no', 'neutral_no']

# Reset the index to convert 'category' from index to a regular column
    sentiment_counts.reset_index(inplace=True)

# Print the resulting DataFrame
    print(sentiment_counts)
    '''





