import pickle
from progressbar import ProgressBar
from pymongo import MongoClient
from collections import OrderedDict


client = MongoClient()
db = client.legislation
bills = db.bills

cursor = bills.find({}, {"text_versions": 1, "bill_id": 1, "official_title": 1}, no_cursor_timeout=True)

bill_dict = OrderedDict()

pbar = ProgressBar(maxval=cursor.count()).start()
for i, selection in enumerate(cursor):
    text = selection['text_versions'].itervalues().next()
    bill_dict.update({selection['bill_id']: text})
    pbar.update(i)
pbar.finish()

with open("all_bills.pickle", mode='wb') as f:
        pickle.dump(bill_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
