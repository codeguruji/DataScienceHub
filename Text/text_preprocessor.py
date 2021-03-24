import re, os
import unicodedata
import codecs
# import HTMLParser
import nltk
import csv
import spacy
import en_core_web_sm
from spellchecker import SpellChecker
import enchant #english dictionary
import us #US address parsing
from gensim.corpora.textcorpus import STOPWORDS
from gensim.models.phrases import Phrases
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.collocations import *
from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction.text import CountVectorizer
from contractions_dict import contractions_dict

## constants
contractions_dict = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "iit will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

# general purpose functions
def strip_accents_unicode(text):
    '''cleans the accents on some characters'''
    if not text:
        return ''
    return ''.join([c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)])
    
def remove_html(text):
    '''removes the html tags, useful when processing data extracted from webpages'''
    if not text:
        return ''
	return re.sub(r'( ?\.+ )+', ' . ', re.sub(r'<[^>]*>', ' . ', text))
    
def join_urls(text, url_pattern):
    '''convert the urls into a token (eg. wwwfacebookcom)'''
    if not text:
        return ''
	m = re.search(url_pattern, text)
	while m:
		text = re.sub(url_pattern, m.group(3).replace("http://","").replace(".",""), text)
		m = re.search(url_pattern, text)
	return text
    
def url_handler(text, url_pattern, handle_with):
    '''finds and replaces the url according to handle_with parameter'''
    if not text:
        return ''
    m = re.search(url_pattern, text)
    
    while m:
        if handle_with == "join":
            text = re.sub(url_pattern, m.group(3).replace("http://","").replace(".",""), text)
        elif handle_with == "remove":
            text = re.sub(url_pattern, " ", text)
        elif handle_with == "replace":
            text = re.sub(url_pattern, "_url_", text)
        m = re.search(url_pattern, text)
    
    return text
    
def join_compound_words(text, compound_pattern, join_with="_"):
    if not text:
        return ''
	m = re.search(compound_pattern, text)
	while m:
		text = re.sub(m.group(0), m.group(0).replace("-",join_with), text)
		m = re.search(compound_pattern, text)
	return text

def symspell_checker(text):
    from symspellpy.symspellpy import SymSpell
    spell = SymSpell()
    spell.load_dictionary(r"frequency_dictionary_en_82_765.txt", 0, 1)
    spell.load_bigram_dictionary(r"frequency_bigramdictionary_en_243_342.txt", 0, 2)
    result = spell.lookup_compound(text, 2)
    for r in result:
        return r.term
    return text    

def contraction_handler(text):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def load_stanford_ner_tagger(stanford_ner_path):
    stanford_ner = StanfordNERTagger(os.path.join(stanford_ner_path,"classifiers/english.all.3class.distsim.crf.ser.gz"), 
											os.path.join(stanford_ner_path,"stanford-ner.jar"))
	stanford_ner._stanford_jar = stanford_ner_path+"stanford-ner.jar:"+stanford_ner_path+"lib/*"
    
    return stanford_ner

def ner_handler_stanford(tagged_corpus, handle_with):
    normalized_corpus = []
    for tagged_doc in tagged_corpus:
        normalized_doc = ""
        current_ner = []
        tags = {}
        for token, tag in enumerate(tagged_doc):
            normalized_doc = normalized_doc + token + " "
            if current_ner:
                if tag == "O" or (tag != "O" and tag != current_ner[-1][1]):
                    tags[' '.join([t for t,_ in current_ner])] = current_ner[0][1]
                    current_ner = []
                if tag != "O":
                    current_ner.append((token, tag))
        if current_ner:
            tags[' '.join([t for t,_ in current_ner])] = current_ner[0][1]
        
        for token, tag in tags.items():
            if handle_with == "replace":
                normalized_doc = normalized_doc.replace(token, "_ner_")
            elif handle_with == "replace_ner":
                normalized_doc = normalized_doc.replace(token, "_{0}_".format(tag.lower()))
            elif handle_with == "join":
                normalized_doc = normalized_doc.replace(token, token.replace(" ", "_"))
        normalized_corpus.append(normalized_doc.strip())
    
    return normalized_corpus
    
def ner_handler_spacy(tagged_corpus, handle_with):
    normalized_corpus = []
    for tagged_doc in tagged_corpus:
        normalized_doc = tagged_doc.text
        for entity in tagged_doc.ents:
            if handle_with == "replace":
                normalized_doc = normalized_doc.replace(entity.text, "_ner_")
            elif handle_with == "replace_ner":
                if entity.label_ in ["PERSON", "ORG", "GPE"]:
                    normalized_doc = normalized_doc.replace(entity.text, "_{0}_".format(entity.label_.lower()))
            elif handle_with == "join":
                normalized_doc = normalized_doc.replace(entity.text, entity.text.replace(" ", "_"))
            else: #remove
                normalized_doc = normalized_doc.replace(entity.text, "")
            
        normalized_corpus.append(normalized_doc)
    return normalized_corpus

def correct_spelling(word, spelling_model=None):
    if not spelling_model:
        spelling_model = SpellChecker()
    if "_" in word:
        return word
    else:
        return spelling_model.correction(word)

def tokenize(corpus):
    tokens = []
    for sent in corpus:
        words = word_tokenize(sent)
        tokens.append(words)
    return tokens

def stem_sent(sent, stemmer):
    tokens = word_tokenize(sent)
    for d in range(len(tokens)):
        if re.match("\w+\_\w+", tokens[d]):
            toks = []
            for t in tokens[d].split("_"):
                toks.append(stemmer.stem(t))
            tokens[d] = "_".join(toks)
        else:
            tokens[d] = stemmer.stem(tokens[d])
    return " ".join(tokens)

def lemmatize_sent(sent, lemmatizer):
    tokens = word_tokenize(sent)
    for d in range(len(tokens)):
        if re.match("\w+\_\w+", tokens[d]):
            toks = []
            for t in tokens[d].split("_"):
                toks.append(lemmer.lemmatize(t))
            tokens[d] = "_".join(toks)
        else:
            tokens[d] = lemmer.lemmatize(tokens[d])
    return " ".join(tokens)
    
def lemmatize_sent_spacy(sent, nlp, pos_keep = [""]):
    doc = nlp(sent)
    result = []
    for d in doc:
        if d.pos_ in pos_keep:
            result.append(d.lemma_)
    return " ".join(result).replace(" _ ", "_")

def get_phrases_corpus(corpus, ngram_range=(2,3), delimiter=b"_"):
    #tokenize the corpus
    corpus_tokens = []
    for d in corpus:
        corpus_tokens.append(word_tokenize(d))
        
    if ngram_range == (1,2):
        ngram_range = (2,2)
	# avoid 
	ngram_range = (min(ngram_range), max(ngram_range))
	# first get the bigram
	phraser = Phrases(corpus_tokens, min_count=5, threshold=10, delimiter=delimiter)
	for _ in range(ngram_range[1]-ngram_range[0]):
		phraser = Phrases(phraser[corpus_tokens], min_count=3, delimiter=delimiter)
	phrases = phraser[corpus_tokens]
	
    for d in range(len(corpus)):
        corpus[d] = " ".join(phrases[d])
    return corpus, phraser
        
def corpus_tagger_stanford(corpus, model):
    ## split into sentences
    corpus_sentences = []
    sent_to_doc_map = {}
    sent_number = 0
    for d in range(len(corpus)):
        for sent in tokenize.sent_tokenize(corpus[d]):
            corpus_sentences.append(tokenize.word_tokenize(sent))
            sent_to_doc_map[sent_number] = d
            sent_number += 1
    tagged_sents = stanford_ner.tag_sents(corpus_sentences)
    tagged_corpus = []
    current_doc_no = 0
	current_doc = []
    for s in range(tagged_sents):
        doc_no = sent_to_doc_map[s]
        if doc_no == current_doc_no:
            current_doc += tagged_sents[s]
        else:
            tagged_corpus.append(current_doc)
            current_doc = tagged_sents[s]
            current_doc_no = doc_no
    tagged_corpus.append(current_doc)
    return tagged_corpus
       
def corpus_tagger_spacy(corpus, model):
    for d in range(len(corpus)):
        corpus[d] = model(corpus[d])
        return corpus
        
def space_out_punctuation(text):
    if not text:
        return ''
	text = re.sub(r',\s', ' , ', text)
	text = re.sub(r'\.\.\.\s', ' ... ', text)
	text = re.sub(r'\.\s', ' . ', text)
	text = re.sub(r';\s', ' ; ', text)
	text = re.sub(r':\s', ' : ', text)
	text = re.sub(r'\?\s', ' ? ', text)
	text = re.sub(r'!\s', ' ! ', text)
	text = re.sub(r'"', ' " ', text)
	text = re.sub(r'\'', ' \' ', text)
	text = re.sub(r'\s\(', ' ( ', text)
	text = re.sub(r'\)\s', ' ) ', text)
	text = re.sub(r'\s\[', ' [ ', text)
	text = re.sub(r'\]\s', ' ] ', text)
	text = re.sub(r'-', ' - ', text)
	text = re.sub(r'_', ' _ ', text)
	text = re.sub(r'\n', ' ', text)
	text = re.sub(r'\r', ' ', text)
	text = re.sub(r'\s+', ' ', text)
	return text
    
def basic_preprocessing(text, lowercase=True, remove_punc=True, keep="_"):
    if not text:
        return ''
    if lowercase:
        text = text.lower()
    if remove_punc:
        text = "".join(c for c in text if c not in string.punctuation.replace(keep, ""))
    text = re.sub("\s+", " ", text)
    return text
    
def summarize_document(text, wordcount=150):
    return summarize(document, word_count=wordcount)
    
def get_important_keywords(text):
    return keywords(text).split("\n")

def remove_stopwords(corpus, stopwords):
    for d in range(len(corpus)):
        corpus[d] = [w for w in corpus[d] if d not in stopwords]
    return corpus

def convert_to_bag_of_words(corpus, stopwords, model="freq", max_features=5000):
    '''model: freq, tfidf'''
    print("converting corpus to bag-of-words format...")
    if model == "freq":
        vectorizer = CountVectorizer(input='content', 
                                decode_error="",
                                strip_accents=True,
                                stop_words=stopwords,
                                lowercase=True,
                                max_df=0.8,
                                min_df=2,
                                max_features=max_features)
    elif model == "tfidf":
        vectorizer = TfidfVectorizer(input='content', 
                                decode_error="",
                                strip_accents=True,
                                stop_words=stopwords,
                                lowercase=True,
                                max_df=0.8,
                                min_df=2,
                                max_features=max_features)
	dtm = vectorizer.fit_transform(corpus)  # a sparse matrix
	vocab = vectorizer.get_feature_names() 	# a list
	print("vocabulary size:", len(vocab))
    return dtm, vocab

class TextPreprocessor(object):
    def __init__(self, decode_error="strict", strip_accents='unicode', lowercase=True, contractions=True,\
                    ignore_list=[], stopwords=None, remove_html=True, treat_urls="join", extract_phrases=True,\
                    treat_ner="replace_ner", lemmatize=False, stemming=True, spellcheck=False, tokenize=True,\
                    join_char="_", 
                    ):
        '''A comprehensive text pre-processing class
        ...
        Attributes
        ----------
        contractions: bool
            whether to decontract words like you're -> you are
        
        ignore_list: list
            list of characters to remove from the text
            
        stopwords: set
            set of words to be considered as stopwords, if None then the stopwords won't be removed
            
        remove_html: bool
            remove html tags important for the text data scrapped from web pages
            
        treat_urls: str: 
            how should be the urls present in the text be treated
            takes values from (join, remove, replace), if selected as replace: urls will be replaced with _url_
        
        extract_phrases: extracts most common phrases from the text and joins them with a predefined character
        
        treat_ner: string
            extracts and treats named entities from the text
            replace: the entities will be replaced by a common token (_ner_)
            replace_ner: entities will be replaced by corresponding named entity eg. _person_, _location_ etc.
            join: entities token will be joined together by a joining character
            None: doesn't extract NER
            
        lemmatize: lemmatize the tokens
        
        stemming: tokens are stemmed
        
        spellcheck: should the spelling be corrected
        
        tokenize: text will be returned as a list of tokens
        
        Methods
        -------
        
        Returns
        -------
        '''
        self.lowercase = lowercase
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.contractions = contractions
        self.ignore_list = ignore_list
        self.stopwords = stopwords
        self.remove_html = remove_html
        self.treat_urls = treat_urls
        self.extract_phrases = extract_phrases
        self.treat_ner = treat_ner
        self.lemmatize = lemmatize
        self.stemming = stemming
        self.spellcheck = spellcheck
        self.tokenize = tokenize
        self.join_char = join_char
        self.compound_pattern = re.compile(r'\w+(\-\w+)+') #here-there
        self.stanford_ner = load_stanford_ner_tagger("stanford_ner_path")
        self.spell_checker = SpellChecker()
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        disable = ['parser']
        if self.treat_ner == None:
            disable.append('ners')
        self.nlp = spacy.load('en_core_web_sm', disable=disable)
    
    def clean_corpus(self, corpus):
        '''cleans a corpus and returns list of clean text'''
        print("cleaning corpus with {0} documents.".format(len(corpus)))
        
        ## first pass of the corpus to remove some noise from the documents
        for d in range(corpus):
            if self.remove_html: # remove html
                corpus[d] = remove_html(corpus[d])
            if self.strip_accents: #strip accents
                corpus[d] = strip_accents_unicode(doc)
            if self.treat_urls: #extract and treat urls
                corpus[d] = url_handler(corpus[d], self.url_pattern, self.treat_urls)
                        
        ## Before normalizing the documents, treat NER                
        if self.treat_ner: #ner handle
            tagged_corpus = corpus_tagger_spacy(corpus, self.nlp)
            corpus = ner_handler_spacy(tagged_corpus, self.treat_ner)

        ## second pass through the corpus
        for d in range(corpus):
            if self.contractions: #expand contractions
                corpus[d] = contraction_handler(corpus[d])
                
            #join compound words
            corpus[d] = join_compound_words(corpus[d], self.compound_pattern, self.join_char)
            
            #spell check
            if self.spellcheck:
                corpus[d] = " ".join([correct_spelling(tok, spell_checker) for tok in word_tokenize(corpus[d])])
            
            #basic pre-processing
            corpus[d] = basic_preprocessing(corpus[d], True, True) #lowercase
            
            #lemmatize
            if self.lemmatize:
                #corpus[d] = lemmatize_sent(corpus[d], self.lemmatizer)
                corpus[d] = lemmatize_sent_spacy(corpus[d], self.nlp)
            elif self.stem: #stemming
                corpus[d] = stem_sent(corpus[d], self.stemmer)
            
            
        #get n-grams
        if self.extract_phrases:
            corpus, self.phraser = get_phrases_corpus(corpus, delimiter=str.encode(self.join_char))
        
    def clean_doc(self, doc):
        '''cleans a piece of text'''
        if self.remove_html: # remove html
            doc = remove_html(doc)
        if self.strip_accents: #strip accents
            doc = strip_accents_unicode(doc)
        if self.treat_urls: #extract and treat urls
            doc = url_handler(doc, self.url_pattern, self.treat_urls)
            
        if self.treat_ner: #ner handle
            tagged_corpus = corpus_tagger_spacy([doc], self.nlp)
            doc = ner_handler_spacy(tagged_corpus, self.treat_ner)[0]
            
        doc = contraction_handler(doc)
        doc = join_compound_words(doc, self.compound_pattern, self.join_char)
        
        if self.spellcheck:
            doc = " ".join([correct_spelling(tok, spell_checker) for tok in word_tokenize(doc)])
            
        doc = basic_preprocessing(doc, True, True) #lowercase
        
        if self.lemmatize:
            #doc = lemmatize_sent(doc, self.lemmatizer)
            doc = lemmatize_sent_spacy(doc, self.nlp)
        elif self.stem: #stemming
            doc = stem_sent(doc, self.stemmer)
            
        if self.extract_phrases and self.phraser:
            doc = " ".join(self.phraser[doc.split(" ")])
            
        return doc