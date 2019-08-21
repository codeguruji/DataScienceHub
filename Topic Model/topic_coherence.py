import pandas as pd
import numpy as np
import time
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

## Load the data
data = pd.read_csv(r"transcript_segment_data_clean1.csv")
corpus = data["transcript_clean"].values
corpus_tokens = [d.split(" ") for d in corpus]

## Vectorize the corpus
cv = CountVectorizer(stop_words="english", min_df=5, max_df=0.4, max_features=5000, ngram_range=(1,1))
dtm = cv.fit_transform(corpus)

features = np.array(cv.get_feature_names())
id2token = dict(zip(range(len(features)), features))
token2id = dict(zip(features, range(len(features))))

## Create a gensim dictionary
dictionary = Dictionary()
dictionary.id2token = id2token
dictionary.token2id = token2id


## Train LDA models with different count of topics
topic_counts = [20, 30, 40, 50, 70, 100, 120, 150]

def get_topn_words(lda_model, features, topn = 20):
    topics = lda_model.components_
    topic_words = []
    for topic_num, topic_weights in enumerate(topics):
        top_words = topic_weights.argsort()[::-1][:topn]
        topic_words.append(list(features[top_words]))
    return topic_words

def get_coherence_lda_models(topic_counts, dtm, features, corpus, dictionary):
    '''returns a list containing coherence values by training LDA model for different topic counts'''
    lda_models_coherence = []
    for num_topics in topic_counts:
        t_start = time.time()
        print("Training model for {0} topics...".format(num_topics))

        ## Train a LDA model
        lda = LatentDirichletAllocation(n_components=num_topics, n_jobs=-1)
        lda = lda.fit(dtm)

        ## Obtain the topics as a list of list of words
        topic_words = get_topn_words(lda, features, 20)

        ## Get the combined coherence for the topics
        cm = CoherenceModel(topics = topic_words, texts=corpus, dictionary=dictionary, coherence="c_v", processes=-1)
        coherence = cm.get_coherence()
        lda_models_coherence.append(coherence)

        t_end = time.time()
        print("time taken: ", t_end-t_start)
    
    return lda_models_coherence

coherences = get_coherence_lda_models(topic_counts, dtm, features, corpus_tokens, dictionary)

import matplotlib.pyplot as plt

plt.plot(range(len(coherences)), coherences, color="r", linewidth=4)
plt.xticks(range(len(coherences)), topic_counts)
plt.show()  
