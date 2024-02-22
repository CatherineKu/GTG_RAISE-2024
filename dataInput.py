#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:32:55 2024

@author: chuhanku
"""

#NLP 统计词语多少
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Dataset_10k (1).csv')

#print(df['translated_title'])


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
print(bow_df["ai"])

# Save the bag-of-words DataFrame to a CSV file
#bow_df.to_csv('bag_of_words.csv', index=False)
# Save the bag-of-words DataFrame to a CSV file
# bow_df.to_csv('bag_of_words.csv', index=False)


