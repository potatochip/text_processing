import nltk
import string
from progressbar import ProgressBar
from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

client = MongoClient()
db = client.legislation
bills = db.bills

token_dict = {}
stemmer = PorterStemmer()
pbar = ProgressBar()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def preprocess():
    # get documentsfrom db
    # iterate through each document and convert text to lowercase and remove punctuation
    text = pass  #bill_text for each document
    lowers = text.lower()
    no_punctuation = lowers.translate(None, string.punctuation)
    token_dict[billid] = no_punctuation

preprocess()
vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
docs = pbar(vectorizer.fit_transform(token_dict.values()))