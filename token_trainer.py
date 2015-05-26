from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
from pymongo import MongoClient
from progressbar import ProgressBar


client = MongoClient()
db = client.legislation
bills = db.bills

# set custom parameters
trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.INCLUDE_ABBREV_COLLOCS = True
extra_collocations = {(u'sec.', u'##number##.')}

# Read in training corpus
# cursor = bills.find({}, {"text_versions": 1})
cursor = bills.find({}, {"text_versions": 1}).limit(1000)

# Train trainer
pbar = ProgressBar(maxval=cursor.count()).start()
for i, line in enumerate(cursor):
    text = line['text_versions'].itervalues().next()
    trainer.train(text, finalize=False, verbose=False)
    pbar.update(i)
pbar.finish()

print "Finalizing training..."
trainer.finalize_training(verbose=True)
print "Training done."

# Include custom parameters
params = trainer.get_params()
params.collocations = params.collocations | extra_collocations

with open('sentence_tokenizer_params.pickle', 'wb') as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
print "Params: %s" % repr(params)

# Create tokenizer
tokenizer = PunktSentenceTokenizer(params)

# Dump pickled tokenizer
with open("sentence_tokenizer.pickle", mode='wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
