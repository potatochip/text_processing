from progressbar import ProgressBar
from pymongo import MongoClient

import nltk
import re
import mpld3
import random

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


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


client = MongoClient()
db = client.legislation
bills = db.bills
votes = db.votes
combo = db.combo


# Read in training corpus
cursor = combo.find({"subjects_top_term": {"$exists": "true"}}, {"_id": 0},
                     no_cursor_timeout=True, limit=2000)
titles, texts, categories, congress, bill_type, number, subject_top, subjects, sponsor,\
    requires, result = zip(*[(i['official_title'], i["text_versions"].itervalues().next(),
                                i["category"], i["congress"], i["bill_type"], i["number"],
                                i["subjects_top_term"], i["subjects"], i["sponsor"], i["requires"],
                                i["result"]) for i in cursor])
bills = {'title': titles, 'vote': result, 'text': texts, 'sponsor': sponsor,
            "subject_top": subject_top, "subjects": subjects, 'category': categories,
            'congress': congress, 'bill_type': bill_type, 'number': number, 'requires': requires}
df = pd.DataFrame(bills, columns=['vote', 'title', 'sponsor', 'subject_top',
                                    'subjects', 'category', 'requires', 'text'])
df['Passed'] = pd.get_dummies(df['vote'])['Passed']

# get sample
# df = random.sample(df, 2000)
X = df.drop('Passed', axis=1)
y = df.Passed
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")


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
            combo.insert_one(bill_data)
        except:
            print("Can't find bill {}-{}-{} for known vote".format(session, billtype, number))

        pbar.update(i)
    pbar.finish()


def tokenize(text, stemmed=True):
    # first tokenize by sentence, then by word
    # to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word
                in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    if stemmed:
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems
    else:
        return filtered_tokens


def multinomial_model():
    # # uncomment to save, comment to load model
    # text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.8, max_features=200000, min_df=0.2, ngram_range=(1,3), use_idf=True)),
    #                     ('tfidf', TfidfTransformer()),
    #                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
    #                                             alpha=1e-3, n_iter=5, random_state=42)), ])
    # text_clf = text_clf.fit(X_train.text, y_train)
    # joblib.dump(text_clf, 'model.pkl')

    text_clf = joblib.load('model.pkl')

    # # predict on model
    # docs_new = X_test.text
    # doc_titles = X_test.title
    # X_new_tfidf = tfidf_vectorizer.transform(doc_titles)
    # predicted = clf.predict(X_new_tfidf)
    # for bill, vote in zip(docs_new, predicted):
    #     print(bill)
    #     print((lambda x: 'Passed' if x == 1 else 'Failed')(vote))

    # docs_test = X_test.text
    predicted = cross_val_score(text_clf, X.text, y)
    # predicted = text_clf.predict(docs_test)
    print predicted
    # print("Accuracy of {}".format(np.mean(predicted == y_test)))
    print df.vote.value_counts()


###################
# define a pipeline combining a text feature extractor with a simple classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__max_n': (1, 2),  # words or bigrams
    'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block
    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(X.text, y)
    print "done in %0.3fs" % (time() - t0)
    print
    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])


# get_voted_bills()
# multinomial_model()
