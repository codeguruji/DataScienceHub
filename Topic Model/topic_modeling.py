import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.sklearn
import pickle
%matplotlib inline
import re

from gensim.corpora import STOPWORDS

num_topics = 25
num_words_in_topic = 25

stop = set(stopwords.words('english'))
