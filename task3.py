#task3
#import packages
from itertools import count
from modulefinder import packagePathMap
import sqlite3
from string import punctuation
import pandas as pd
import re
import nltk
import operator
from matplotlib import pylab
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import math
import numpy as np

#import passage data and preprocessing from task2
data = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names= ['qid','pid','query','passage'])
lemmatizer  = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
data["passage"] = data["passage"].str.replace('[^\w\s]','')
data["passage"] = data["passage"].str.replace("\d+", "")
data["passage"] = data["passage"].apply(lambda x : str.lower(x))
data["passage"] = data["passage"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
data["passage"] = data["passage"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data["query"] = data["query"].str.replace('[^\w\s]','')
data["query"] = data["query"].str.replace("\d+", "")
data["query"] = data["query"].apply(lambda x : str.lower(x))
data["query"] = data["query"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
data["query"] = data["query"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data_new = data[["pid","passage"]]

#import query data and preprocessing
df = pd.read_csv('test-queries.tsv', sep='\t', header=None, names= ['qid','query'])
df["query"] = df["query"].str.replace('[^\w\s]','')
df["query"] = df["query"].str.replace("\d+", "")
df["query"] = df["query"].apply(lambda x : str.lower(x))
df["query"] = df["query"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
df["query"] = df["query"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df
#get the inverted index from task2
def generate_inverted_index(data: list):
    inv_idx_dict = {}
    for index, doc_text in data.values:
        for word in doc_text.split():
            if word not in inv_idx_dict.keys():
                inv_idx_dict[word] = [index]
            elif index not in inv_idx_dict[word]:
                inv_idx_dict[word].append(index)
    return inv_idx_dict
inverted_index = generate_inverted_index(data_new)
inverted_index

#function of calculate the TF from a dataframe:
def computeTF(data):
    counts = {}
    for pid,passage in data.values:
        p = passage.split()
        nwords = len(p)
        words = set(p)
        for word in words:
            if word not in counts:
                counts[word] = [p.count(word)/nwords]
            else:
                counts[word].extend([p.count(word)/nwords])
    return counts
pass_TF = computeTF(data_new)
pass_TF


# function of compute IDF and get the IDF of the passage
def computeIDF(InvertedIndex, documentCount):
    IDF = {}
    for key, pid in InvertedIndex.items():
        IDF[key] = math.log10(float(documentCount) / len(InvertedIndex[key]))

    return IDF


n = len(data_new)
pass_IDF = computeIDF(inverted_index, n)
pass_ID

#function of getting a TFIDF of passage
def TFIDF(TF,IDF):
    words = list(IDF.keys())
    TFIDF = {}
    for word in TF:
        if word in words:
            TFIDF[word] = [i*IDF[word] for i in TF[word]]
        else:
            TFIDF[word]=[0]

    return TFIDF

pass_TFIDF = TFIDF(pass_TF,pass_IDF)
pass_TFIDF

##get the TF for query
query_TF = computeTF(df)
query_TF
##get the TFIDF for query
query_TFIDF = TFIDF(query_TF,pass_IDF)
query_TFIDF