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

from gensim.parsing.preprocessing import STOPWORDS

num_topics = 25
num_words_in_topic = 25

stop = set(stopwords.words('english'))
stop.update(STOPWORDS)

## Load the data
data = pd.read_csv("data.csv")
corpus = data["text"].values

## vectorize the text
vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1,2), \
                            min_df=5, max_df=0.5, token_pattern=r'[a-zA-Z0-9_]{3,}')
dtm = cv.fit_transform(corpus)
features = np.array(cv.get_feature_names())

## Fit the topic model
lda = LatentDirichletAllocation(n_components=num_topics, learning_method='batch',
                                batch_size=128, max_iter=10, n_jobs=-1, random_state=42)
lda = lda.fit(dtm)

print("model perplexity:", lda.perplexity(dtm))

## Get the topic associations for the corpus documents
doc_topics = lda.transform(dtm)
topic_names = ["Topic_{0}".format(t) for t in range(num_topics)]
document_names = ["Doc_{0}".format(i) for i in range(len(corpus))]

#get the document-topic association probabilities
doc_topics_df = pd.DataFrame(data = np.round(doc_topics, 2), columns = topic_names, index=document_names)

#Dominant topic in a document
doc_topics_df['dominant_topic_association'] = np.max(doc_topics_df.values, axis=1)
doc_topics_df['dominant_topic'] = np.argmax(doc_topics_df.values, axis=1)

#topic distribution
topic_distribution = doc_topics_df['dominant_topic'].value_counts().reset_index(name="Num Documents")

#get the topic word distribution
## Top words for every topic
topic_words = lda.components_
topic_top_words = []
for topic_num, topic_weights in enumerate(topic_words):
    top_words = topic_weights.argsort()[::-1][:num_words_in_topic]
    topic_top_words.append(feature_names[top_words])

topic_top_words = pd.DataFrame(topic_top_words)
topic_top_words.columns= ["Word" + str(i) for i in range(num_words_in_topic)]
topic_top_words.index= ["Topic" + str(i) for i in range(num_topics)]


#visualize LDA model
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, dtm, vectorizer, mds='tsne')
with open("topic_visualization.html", "w") as f:
    pyLDAvis.save_html(panel, f)
