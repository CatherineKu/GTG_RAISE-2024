#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:09:47 2024

@author: chuhanku
"""

import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Dataset_10k (1).csv')

df_cut = df.iloc[:50]

import requests
from bs4 import BeautifulSoup
from googletrans import Translator
from summarizer import Summarizer
import pandas as pd
import time

# Load your DataFrame 'df' containing the 'link' column

'''
# Function to fetch article text from a given URL
def fetch_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Example: Extracting text from <p> tags, adjust according to the structure of the website
    article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return article_text

'''
def fetch_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check if soup is not None
        if soup is not None:
            # Extract text from <p> tags
            article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return article_text
        else:
            print(f"Error: Failed to fetch content from {url}. Soup object is None.")
            return None

    except Exception as e:
        print(f"Error: Failed to fetch content from {url}: {e}")
        return None

'''
# Function to translate text if it's not in English
def translate_if_not_english(text):
    translator = Translator()
    if translator.detect(text).lang != 'en':
        translated_text = translator.translate(text, src='auto', dest='en')
        return translated_text.text
    else:
        return text
'''

def translate_if_not_english(text, language):
    if language != 'en':
        translator = Translator()
        translated_text = translator.translate(text, src='auto', dest='en')
        return translated_text.text
    else:
        return text


# List to store the summaries for each link
summaries = []

import requests
from requests.exceptions import Timeout

pip install sumy


# Define the timeout value (in seconds)
REQUEST_TIMEOUT = 10
MIN_REQUEST_INTERVAL = 5
for i, (link, language) in enumerate(zip(df_cut['link'], df_cut['language'])):
    try:
        # Fetch the article text from the link
        article_text = fetch_article_text(link)

        # Translate the text if it's not in English
        article_text_en = translate_if_not_english(article_text, language)

        # Use the translated or original article text as src_text
        src_text = [article_text_en] if article_text_en else [article_text]

        # Generate the summary
        summary = Summarizer(src_text)  # You can adjust the min_length and max_length parameters

       # Debugging print statements

        # Convert summary to text
        summary_text = ' '.join(summary)
        print("Summary text:", summary_text)

        # Convert summary to bag of words
        tokens = nltk.word_tokenize(summary_text)
        bag_of_words = Counter(tokens)
        
        print("Bag of words:", bag_of_words)

        print(f"Processed link {i + 1}/{len(df_cut)}")

        # Check if it's not the last iteration
        if i + 1 < len(df_cut):
            time.sleep(MIN_REQUEST_INTERVAL)

    except TypeError as e:
        print(f"Error processing link '{link}': {e}")
        # If an error occurs, append "0" to the list of summaries
        summaries.append("0")
    except Exception as e:
        print(f"Unknown error processing link '{link}': {e}")
        # If an error occurs, append "0" to the list of summaries
        summaries.append("0")

# Add the summaries to the DataFrame as a new column
df_cut['summary'] = bag_of_words
                                                                                                                        
# Output the DataFrame with summaries
df_cut.to_csv('df_cut.csv', index=False)