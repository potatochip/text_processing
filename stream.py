'''
streams tweets from legislators to mongodb
'''

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import cnfg
import json
from pymongo import MongoClient

config = cnfg.load(".twitter_config")

auth = OAuthHandler(config['consumer_key'], config['consumer_secret'])
auth.set_access_token(config['access_token'], config['access_token_secret'])

client = MongoClient()
db = client.legislation
tweets = db.tweets


class StreamListener(StreamListener):
    def on_connect(self):
        print("You're connected to the streaming server.")

    def on_error(self, status_code):
        print('Error: ' + repr(status_code))
        return True  # Don't kill the stream

    def on_timeout(self):
        return True  # Don't kill the stream

    def on_data(self, data):
        datajson = json.loads(data)
        tweets.insert(datajson)
        print("got one!")


def get_handles():
    legislators = db.legislators
    twitter_ids = legislators.find({'twitter': {'$ne': ''}}, {'twitter': 1})
    return [i['twitter'] for i in twitter_ids]


if __name__ == '__main__':
    l = StreamListener()
    stream = Stream(auth, l)
    stream.filter(track=get_handles()[:400])
