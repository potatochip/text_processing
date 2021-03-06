import tweet_dumper
from pymongo import MongoClient
import os
import time
from os import listdir
from os.path import isfile, join
import csv
import json

client = MongoClient()
db = client.legislation


def get_handles():
    legislators = db.legislators
    twitter_ids = legislators.find({'twitter': {'$ne': ''}}, {'twitter': 1})
    return [i['twitter'] for i in twitter_ids]


def mongo_save():
    tweets = db.tweets
    mypath = os.getcwd()
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    tweet_files = [i for i in onlyfiles if i.endswith('csv')]

    fieldnames = ("id", "created_at", "text", "retweets", "favorites")
    for i in tweet_files:
        csvfile = open(i, 'r')
        reader = csv.DictReader(csvfile, fieldnames)
        reader.next()
        for row in reader:
            save_me = json.loads(json.dumps(row))
            save_me.update({"handle": i[:-11]})
            tweets.insert(save_me)
        print("working on {0}".format(i))


def get_tweets():
    for index, handle in enumerate(get_handles()):
        print("hey, {0}!".format(handle))
        if not os.path.isfile(handle+"_tweets.csv"):
            if (index + 1) % 10 == 0: time.sleep(900)
            try:
                tweet_dumper.get_all_tweets(handle)
            except:
                print("** problem with {0}".format(handle))

mongo_save()
