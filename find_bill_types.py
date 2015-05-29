import nltk
import string
from pymongo import MongoClient
from progressbar import ProgressBar

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics


client = MongoClient()
db = client.legislation
bills = db.bills

cursor = bills.find({}, {"text_versions": 1, "bill_id": 1, "official_title": 1}, no_cursor_timeout=True)

pbar = ProgressBar(maxval=cursor.count()).start()
for i, selection in enumerate(cursor):
    text = selection['text_versions'].itervalues().next()
    pbar.update(i)
pbar.finish()
