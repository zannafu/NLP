#IMPORT NLTK PACKAGES
from itertools import count
from modulefinder import packagePathMap
import sqlite3
from string import punctuation
import pandas as pd
from pytest import FixtureRequest
import re
import nltk
import operator
from matplotlib import pylab
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from matplotlib import pylab
import math

#IMPORT txt doc:
with open("passage-collection.txt",encoding="utf8") as text_file:
    passage = text_file.read()

#preprocessing
#1. remove puctations & digits
passage = re.sub(r'[^\w\s]',"",passage)
passage = re.sub("\d+", "", passage)
#2. tokenization
words = nltk.word_tokenize(passage.lower())
#3.lemmatisation
lemmatizer  = WordNetLemmatizer()
bag_of_words = []
for word in words:
    bag_of_words.append(lemmatizer.lemmatize(word))

# Python code to find frequency of each word
def word_by_freq(bag):
    freqs = dict()
    for string in bag:
        words = string.split()
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] =1
    return freqs

counts = word_by_freq(bag_of_words)
sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1],reverse=True))
rank = list(range(1,len(counts)+1))
freqs = [freq/len(bag_of_words) for (word, freq) in sorted_counts.items()]


#plot of frequency VS ranking
Y = [1/k for k in rank]
c = sum(Y)
Y_new = [y/c for y in Y]
plt.plot(rank,Y_new,linestyle='dashed',label = "theory(Zipf's law)")
plt.plot(rank,freqs, label = "data")
plt.xlabel("Term frequency ranking")
plt.ylabel("Term prob of occurrence")
plt.legend(loc='upper right')
plt.show()
#here f(r) = 1/c*r, where c is the constant to make the distribution of f(r) normalised
#p(r)=1/C*r**α, That is, the frequency of rth most frequent word in a language is a power law with exponent α; 
#the canonical Zipf’s law is obtained when α=1. 
# C is a constant that makes the distribution add up to 1.


#log-log plot of frequency VS ranking
pylab.loglog(rank, freqs, label='data') 
pylab.loglog(rank,Y_new,linestyle = "dashed",label = "theory(Zipf's law)")
pylab.xlabel('log(rank)')
pylab.ylabel('log(freq)')
pylab.legend(loc='lower left')
pylab.show()