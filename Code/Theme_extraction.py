#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:22:25 2024

@author: chuhanku
"""

import pandas as pd


df1 = pd.read_csv("df_cut_1.csv")
df2 = pd.read_csv("df_cut_2.csv")
df3 = pd.read_csv("df_cut_3.csv")
df4 = pd.read_csv("df_cut_4.csv")
df5 = pd.read_csv('df_cut_101_190.csv')
df6 = pd.read_csv('df_cut_301.csv')
df7 = pd.read_csv('df_cut_351.csv')
df8 = pd.read_csv('df_cut_401_500.csv')

df9 = pd.read_csv('df_cut_501_600.csv')
df10 = pd.read_csv('df_cut_601_700.csv')
df11 = pd.read_csv('df_cut_701_800.csv')
df12 = pd.read_csv('df_cut_801_900.csv')
df13 = pd.read_csv('df_cut_901_1000.csv')
df14 = pd.read_csv('df_cut_1001_1100.csv')
df15 = pd.read_csv('df_cut_1101_1200.csv')


# Assuming df1 and df2 are your two DataFrames with the same columns
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15], ignore_index=True)



#change all strings to be lower
combined_df['summary']=combined_df['summary'].str.lower()

#get rid of unwanted characters such as punctuation marks
combined_df['summary']=combined_df['summary'].str.replace('[^\w\s]','')

#removing numerals
combined_df['summary']=combined_df['summary'].str.replace('\d+','') 

combined_df['summary']=combined_df['summary'].str.replace('\n',' ').str.replace('\r','')


from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
from gensim.models import LdaModel


# Convert the "summary" column to a list
summaries = combined_df['summary'].tolist()

# Words to remove
words_to_remove = {"ai", "artificial intelligence", "artificial", "gpt", "chatbot", "intelligence","generative chatgpt","says", "said"}

# Tokenize and preprocess the text, excluding specific words
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3 and token not in words_to_remove:  # Remove stopwords, short tokens, and specific words
            result.append(token)
    return result


# Preprocess the summaries
processed_summaries = [preprocess(summary) for summary in summaries]

# Create a dictionary mapping of words to unique ids
dictionary = corpora.Dictionary(processed_summaries)

# Create a bag-of-words corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_summaries]

# Set the number of topics
num_topics = 3

# Train the LDA model
lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print the topics and associated keywords
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx} \nWords: {topic}\n')




