## Use this script to retrain the popular pre-trained models on your custom corpus
## Use 

import os
import re

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn import datasets

import gensim.downloader as api
api.info()  # return dict with info about available models/datasets
api.info("text8")

## If you have downloaded the files then load using the following method
def load_glove_files_local(glove_path):
    #glove_path = "../scoring_file/w2v_glove/glove.6B.300d.txt"
    file_path = os.path.join(os.getcwd(), glove_path)
    glove_file = datapath(os.path.join(os.getcwd(), glove_path))
    tmp_file = get_tmpfile("w2v.txt")
    #convert from glove to word2vec
    _ = glove2word2vec(glove_file, tmp_file)
    #load the keyed vectors
    model_glove = KeyedVectors.load_word2vec_format(tmp_file)
    return model_glove


## Load the glove model trained on 6B wikipedia tokens
model_glove = api.load("glove-wiki-gigaword-300")  # load glove vectors
    
data
model = Word2Vec(size=300, min_count=1)
model.build_vocab(sentences=sentences)
total_examples = model.corpus_count
    
