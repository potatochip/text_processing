from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters, PunktTrainer
import pickle
from pymongo import MongoClient
from progressbar import ProgressBar



client = MongoClient()
db = client.legislation
bills = db.bills

# set custom parameters
punkt_param = PunktParameters()
# punkt_param.abbrev_types = {'sec'}
extra_collocations = {(u'sec.', u'##number##.')}
# extra_abbreviations = {r'/sec\s\d+/i'}
# extra_collocations = {}


# Make a new Tokenizer
# tokenizer = PunktSentenceTokenizer(punkt_param)
tokenizer = PunktSentenceTokenizer()
# tokenizer._params.abbrev_types.update(extra_abbreviations)
# tokenizer._params.collocations.update(extra_collocations)

# Read in trainings corpus
# selection = list(bills.find({}, {"text_versions": 1}).limit(1000))
# selection = list(bills.find({}, {"text_versions": 1}))
cursor = bills.find({}, {"text_versions": 1})
# texts = [i['text_versions'].itervalues().next() for i in selection]
# text = '\n'.join(texts)

trainer = PunktTrainer()

# Train tokenizer
pbar = ProgressBar(maxval=cursor.count()).start()
for i, line in enumerate(cursor):
    text = line['text_versions'].itervalues().next()
    trainer.train(text, finalize=False, verbose=False)
    pbar.update(i)
# tokenizer.train(text, verbose=True)
pbar.finish()

print "Finalizing training..."
trainer.finalize_training(verbose=True)
print "Training done."

params = trainer.get_params()

params.collocations = params.collocations | extra_collocations

with open('sentence_tokenizer_params.pickle', 'wb') as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
print "Params: %s" % repr(params)

tokenizer = PunktSentenceTokenizer(params)

# Dump pickled tokenizer
with open("sentence_tokenizer.pickle", mode='wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
