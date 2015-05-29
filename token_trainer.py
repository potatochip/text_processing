from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
from pymongo import MongoClient
from progressbar import ProgressBar


client = MongoClient()
db = client.legislation
bills = db.bills

trainer = PunktTrainer()

# # set custom parameters
extra_collocations = {(u'sec', u'##number##')}
extra_sentence_starters = {u'(##number##)'}
# extra_abbreviations = {u'U.S.C', u'usc'}

trainer.ABBREV = 0.3
"""cut-off value whether a 'token' is an abbreviation"""

trainer.IGNORE_ABBREV_PENALTY = False
"""allows the disabling of the abbreviation penalty heuristic, which
exponentially disadvantages words that are found at times without a
final period."""

trainer.ABBREV_BACKOFF = 5
"""upper cut-off for Mikheev's(2002) abbreviation detection algorithm"""

trainer.COLLOCATION = 7.88
"""minimal log-likelihood value that two tokens need to be considered
as a collocation"""

trainer.SENT_STARTER = 30
"""minimal log-likelihood value that a token requires to be considered
as a frequent sentence starter"""

trainer.INCLUDE_ALL_COLLOCS = False
"""this includes as potential collocations all word pairs where the first
word ends in a period. It may be useful in corpora where there is a lot
of variation that makes abbreviations like Mr difficult to identify."""

trainer.INCLUDE_ABBREV_COLLOCS = False
"""this includes as potential collocations all word pairs where the first
word is an abbreviation. Such collocations override the orthographic
heuristic, but not the sentence starter heuristic. This is overridden by
INCLUDE_ALL_COLLOCS, and if both are false, only collocations with initials
and ordinals are considered."""
""""""

trainer.MIN_COLLOC_FREQ = 1
"""this sets a minimum bound on the number of times a bigram needs to
appear before it can be considered a collocation, in addition to log
likelihood statistics. This is useful when INCLUDE_ALL_COLLOCS is True."""


# Read in training corpus
# cursor = bills.find({}, {"text_versions": 1}, no_cursor_timeout=True)
# cursor = bills.find({}, {"text_versions": 1}, no_cursor_timeout=True, limit=10000)
cursor = bills.find({"congress": {"$in": ["114", "113"]}}, {"text_versions": 1}, no_cursor_timeout=True)

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
# params.collocations = params.collocations | extra_collocations
# params.sent_starters = params.sent_starters | extra_sentence_starters

with open('sentence_tokenizer_params.pickle', 'wb') as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
print "Params: %s" % repr(params)

# Create tokenizer
tokenizer = PunktSentenceTokenizer(params)

# Dump pickled tokenizer
with open("sentence_tokenizer.pickle", mode='wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
