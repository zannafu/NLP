#task2
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

#import data

data = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names= ['qid','pid','query','passage'])
data

# the same preprocessing to the data in task2 + remove stopwords
lemmatizer  = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
data["passage"] = data["passage"].str.replace('[^\w\s]','')
data["passage"] = data["passage"].str.replace("\d+", "")
data["passage"] = data["passage"].apply(lambda x : str.lower(x))
data["passage"] = data["passage"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
data["passage"] = data["passage"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

###code for inverted index
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