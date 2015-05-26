#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#
# Copyright 2013, 2014 Guenter Bartsch
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#
# train NLTK's punkt sentence segmenter
#

import os
import nltk
import pickle
from os.path import expanduser
import ConfigParser
from pymongo import MongoClient

#
# load config, set up global variables
#

home_path = expanduser("~")

config = ConfigParser.RawConfigParser()
config.read("%s/%s" % (home_path, ".airc"))

# sentences = config.get("speech", "sentences")

trainer = nltk.tokenize.punkt.PunktTrainer()

count = 0

client = MongoClient()
db = client.legislation
bills = db.bills
selection = list(bills.find({}, {"text_versions": 1}).limit(1000))
texts = [i['text_versions'].itervalues().next() for i in selection]
text = '\n'.join(texts)
print text

for line in text:

    sent = line.decode('UTF8').rstrip()

    trainer.train(sent, finalize=False, verbose=False)

    count += 1
    if count % 1000 == 0:
        print "%6d sentences." % count

print
print "Finalizing training..."
trainer.finalize_training(verbose=True)
print "Training done. %d sentences." % count
print

params = trainer.get_params()
print "Params: %s" % repr(params)

tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer(params)
with open("de_punkt.pickle", mode='wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

print "All done."
print
