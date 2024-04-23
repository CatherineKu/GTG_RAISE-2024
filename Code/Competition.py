#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import random
import pandas as pd

df = pd.read_csv("Dataset_10k.csv")

df['translated_title']

from langdetect import detect

#change all strings to be lower
df['translated_title']=df['translated_title'].str.lower()

#get rid of unwanted characters such as punctuation marks
df['translated_title']=df['translated_title'].str.replace('[^\w\s]','')

#removing numerals
df['translated_title']=df['translated_title'].str.replace('\d+','') 

df['translated_title']=df['translated_title'].str.replace('\n',' ').str.replace('\r','')


# In[4]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Assuming df is your DataFrame
# If not, replace it with your DataFrame name

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Apply Vader sentiment analysis to the 'translated_title' column
# and assign it to a new column named 'compound'
df['compound'] = df['translated_title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Assuming 'compound' is the column containing VADER compound scores in your DataFrame
df['vader_sentiment'] = df['compound'].apply(lambda score: 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')

# Display the DataFrame with the new columns
print(df[['translated_title', 'compound', 'vader_sentiment']])


# In[5]:


# Assuming df is your DataFrame
# If not, replace it with your DataFrame name

# Extract rows with negative sentiment
vader_negative_df = df[df['vader_sentiment'] == 'negative']

# Save the extracted DataFrame to a CSV file
vader_negative_df.to_csv('vader_negative.csv', index=False)

# Display the first few rows of the extracted DataFrame
print(vader_negative_df.head())


# In[6]:


# Calculate the count of negative sentiments
negative_count = df[df['vader_sentiment'] == 'negative']['vader_sentiment'].count()

# Calculate the total count of sentiments
total_count = df['vader_sentiment'].count()

# Calculate the ratio of negative sentiments
negative_ratio = negative_count / total_count

# Display the result
print(f"Ratio of negative sentiments: {negative_ratio}")


# In[7]:


# Calculate the count of negative sentiments
positive_count = df[df['vader_sentiment'] == 'positive']['vader_sentiment'].count()

# Calculate the total count of sentiments
total_count = df['vader_sentiment'].count()

# Calculate the ratio of negative sentiments
positive_ratio = positive_count / total_count

# Display the result
print(f"Ratio of positive sentiments: {positive_ratio}")


# In[8]:


# Calculate the count of negative sentiments
neutral_count = df[df['vader_sentiment'] == 'neutral']['vader_sentiment'].count()

# Calculate the total count of sentiments
total_count = df['vader_sentiment'].count()

# Calculate the ratio of negative sentiments
neutral_ratio = neutral_count / total_count

# Display the result
print(f"Ratio of neutral sentiments: {neutral_ratio}")


# In[9]:


import os
import random
import pandas as pd
import numpy as np


df = pd.read_csv('Dataset_10k.csv')

df['translated_title']

from langdetect import detect

#change all strings to be lower
df['translated_title']=df['translated_title'].str.lower()

#get rid of unwanted characters such as punctuation marks
df['translated_title']=df['translated_title'].str.replace('[^\w\s]','')

#removing numerals
df['translated_title']=df['translated_title'].str.replace('\d+','') 

df['translated_title']=df['translated_title'].str.replace('\n',' ').str.replace('\r','')

from textblob import TextBlob

#apply textblob sentiment to our translated column
#and assign it to a new column named polarity
df['polarity'] = df['translated_title'].apply(lambda x: TextBlob(x). sentiment)

#split the list into two and create two new columns
#assign the return values of sentiment function;
#polarity and subjectivity to those columns
sentiment_series = df['polarity'].tolist()

df[['polarity1','subjectivity']]=pd.DataFrame(sentiment_series,
       index=df.index)
df.drop('polarity', inplace=True, axis=1)

from wordcloud import WordCloud
wc = WordCloud( background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

df


# In[10]:


# Assuming df is your DataFrame
negative_rows = df[df['polarity1'] < 0]

# Save the DataFrame to a CSV file
negative_rows.to_csv('negative_sentiment_data.csv', index=False)


# In[11]:


df['label'] = df['polarity1'].apply(lambda x: 1 if x > 0 else 0)

# Display the labeled DataFrame
print(df[['translated_title', 'polarity1', 'label']].head())


# In[12]:


pip install vaderSentiment


# In[13]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Sample topics and associated words with weights
topics = {
    0: {"generative":0.023, "google":0.016, "chatgpt":0.012, "chatbots": 0.009, "time": 0.007, "meta": 0.007, "bard": 0.007, "tool": 0.007, "business":0.007,"open":0.006 },
    
    1: {"generative":0.010 ,"business":0.009 ,"robot": 0.008,"news":0.007 ,"today":0.006 ,"tech":0.005 ,"human": 0.005,"launch": 0.004,"help": 0.004,"world":0.004},
    
    2: {"finance": 0.011,
    "new": 0.010,
    "yahoo": 0.010,
    "business": 0.008,
    "news": 0.008,
    "generative": 0.007,
    "launch": 0.006,
    "ibm": 0.005,
    "world": 0.004,
    "market": 0.004},
    
    3: {"new": 0.022,
    "news": 0.015,
    "time": 0.011,
    "human": 0.008,
    "stock": 0.007,
    "fool": 0.007,
    "motley": 0.007,
    "generative": 0.007,
    "technology": 0.006,
    "robot": 0.005},
    
    4: {"model": 0.019,
    "news": 0.014,
    "language": 0.013,
    "generative": 0.011,
    "launch": 0.008,
    "google": 0.008,
    "new": 0.007,
    "large": 0.006,
    "time": 0.006,
    "data": 0.006}
}

# Merge word weights from all topics
combined_weights = {}
for words_weights in topics.values():
    combined_weights.update(words_weights)

# Set the word cloud parameters
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap="pink")

# Generate word cloud from frequencies
wordcloud.generate_from_frequencies(combined_weights)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Combined Word Cloud for All Topics")
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

