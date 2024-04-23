#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:17:30 2024

@author: chuhanku
"""

# set seeds for reproducability

import pandas as pd
import numpy as np
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('vader_negative.csv')

print(df)

english_df = df[df['language'] == 'en'].copy()

# Display the new DataFrame
print(english_df)


df_cut = english_df.iloc[0:50]

import requests
from bs4 import BeautifulSoup
from googletrans import Translator
from summarizer import Summarizer
import pandas as pd
import time

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
from googletrans import Translator

def translate_if_not_english(text, language):
    if language != 'en':
        translator = Translator()
        try:
            translated_text = translator.translate(text, src='auto', dest='en')
            return translated_text.text
        except Exception as e:
            print(f"Translation failed: {e}")
            return text
    else:
        print("No translation needed for English text")
        return text
'''
    
    


from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


summaries = []

# Define the timeout value (in seconds)
REQUEST_TIMEOUT = 10
MIN_REQUEST_INTERVAL = 5
for i, (link, language) in enumerate(zip(df_cut['link'], df_cut['language'])):
    try:
        print(f"Processing row {i + 1}/{len(df_cut)}...")
        article_text = fetch_article_text(link)
        
        # Check if language is English
        if language == 'en':
            inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        else:
            print(f"Not processing non-English text for link '{link}'")
            summaries.append("0")

        if i + 1 < len(df_cut):
            time.sleep(MIN_REQUEST_INTERVAL)

    except TypeError as e:
        print(f"Error processing link '{link}': {e}") # Print traceback information
        summaries.append("0")
    except Exception as e:
        print(f"Unknown error processing link '{link}': {e}")  # Print traceback information
        summaries.append("0")

df_cut['summary'] = summaries
df_cut.to_csv('df_cut_1.csv', index=False)











