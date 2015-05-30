from progressbar import ProgressBar
from pymongo import MongoClient

import nltk
import re
import pickle
import random
import string

import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import logging

from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from pymongo.cursor import CursorType


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


client = MongoClient()
db = client.legislation


stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

limit = 500  # limit sample size for testing. 

with open('sentence_tokenizer_full.pickle', 'rb') as f:
    custom_tokenizer = pickle.load(f)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=0)
def update_pbar():
    if update_pbar.counter != limit:
        update_pbar.counter += 1
    pbar.update(update_pbar.counter)

def get_voted_bills():
    # creates a new database saving important vote and bill information together
    vote_cursor = db.votes.find({'bill': {'$exists': 'true'}, 'result':
                                {'$in': ['Passed', 'Failed']}}, {'bill': 1, 'result': 1,
                                'requires': 1, 'category': 1,  '_id': 0})

    pbar = ProgressBar(maxval=vote_cursor.count()).start()
    for i, vote in enumerate(vote_cursor):
        session, number, billtype = vote['bill']['congress'], vote['bill']['number'],\
                                    vote['bill']['type']

        bill_data = db.bills.find_one({'bill_type': billtype, 'congress': str(session),
                                        'number': str(number)}, {'congress': 1, 'bill_type': 1,
                                        'number': 1, 'official_title': 1, 'subjects_top_term': 1,
                                        'subjects': 1, 'summary': 1, 'sponsor': 1, 'cosponsors': 1,
                                        'text_versions': 1, '_id': 0}, no_cursor_timeout=True)

        try:
            bill_data.update({'category': vote['category'], 'result': vote['result'],
                                'requires': vote['requires']})
            db.combo.insert_one(bill_data)
        except:
            print("Can't find bill {}-{}-{} for known vote".format(session, billtype, number))
        pbar.update(i)
    pbar.finish()


def get_corpus():
    # Read in training corpus
    print("Getting documents")
    cursor = db.combo.find({}, {"text_versions": 1, "result": 1, "_id": 0},
                         no_cursor_timeout=True, cursor_type=CursorType.EXHAUST)
    df = pd.DataFrame()
    pbar = ProgressBar(maxval=limit).start()
    for index, row in enumerate(cursor):
        if index == limit: break
        df = df.append(row, ignore_index=True)
        pbar.update(index)
    pbar.finish()
    cursor.close()

    # df['Passed'] = pd.get_dummies(df['result'])['Passed']

    # # get sample
    # failed = df[df.Passed == 0]
    # passed = df[df.Passed == 1].ix[random.sample(df[df.Passed == 1].index, failed.Passed.count())]
    # df = passed.append(failed)
    # df = random.sample(df, 1000)

    X = df.text_versions.apply(lambda x: x.itervalues().next())  # choose one text versions
    y = pd.get_dummies(df['result'])['Passed']
    print("\nVotes Breakdown for {} votes".format(df.result.count()))
    print(df.result.value_counts(normalize=True))
    return X, y


def tokenize(text, stemmed=False):

    # words = nltk.word_tokenize(sentences)
    no_punctuation = text.lower().encode('utf-8').translate(None, string.punctuation)
    words = nltk.word_tokenize(no_punctuation)
    custom_stop = ['sec', 'section', ]
    stop_words = stopwords + custom_stop
    stopped = [i for i in words if i not in stop_words]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in stopped:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    if stemmed:
        stems = [stemmer.stem(t) for t in filtered_tokens]
        tokens = stems
    update_pbar()
    return filtered_tokens

def multinomial_model():
    # # uncomment to save, comment to load model
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english',
                                                    max_df=0.8, max_features=200000, min_df=0.2,
                                                    ngram_range=(1, 3), use_idf=True)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                alpha=1e-3, n_iter=5, random_state=42)),
                         ])
    text_clf = text_clf.fit(X_train, y_train)
    joblib.dump(text_clf, 'model.pkl')

    # text_clf = joblib.load('model.pkl')

    # # predict on model
    # docs_new = X_test
    # doc_titles = X_test.title
    # X_new_tfidf = tfidf_vectorizer.transform(doc_titles)
    # predicted = clf.predict(X_new_tfidf)
    # for bill, vote in zip(docs_new, predicted):
    #     print(bill)
    #     print((lambda x: 'Passed' if x == 1 else 'Failed')(vote))


def griddy(pipeline):
    # get best params with cross fold validation for both the feature extraction and the classifier
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # words or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'clf__alpha': (0.00001, 0.000001),
        # 'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1)
    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print "done in %0.3fs" % (time() - t0)
    print
    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])


###################
# define a pipeline combining a text feature extractor with a simple classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
    # ('clf', MultinomialNB()),
])


if __name__ == "__main__":
    X, y = get_corpus()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # # multiprocessing requires the fork to happen in a __main__ protected block
    griddy(pipeline)

    # print("\nCalculating scores")
    # pbar = ProgressBar(maxval=len(y)).start()

    # # full report
    # text_clf = pipeline.fit(X_train, y_train)
    # predicted = text_clf.predict(X_test)
    # print(metrics.classification_report(y_test, predicted))

    # cross validated accuracy score
    # predicted = cross_val_score(pipeline, X, y, n_jobs=1)
    # print("Accuracy of {}".format(np.mean(predicted)))
    # pbar.finish()


# get_voted_bills()
# multinomial_model()
