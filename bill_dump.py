from pymongo import MongoClient
import os
from os import listdir
from os.path import join, isdir, isfile
import json


client = MongoClient()
db = client.legislation
bills = db.bills
votes = db.votes

mypath = os.getcwd() + "/congressional_data/sessions/"


def get_sessions():
    sessions = []
    for f in listdir(mypath):
        if isdir(join(mypath, f)):
            sessions.append(mypath + f)
    return sessions


def remove_dot_key(obj):
    for key in obj.keys():
        new_key = key.replace(".", "")
        if new_key != key:
            obj[new_key] = obj[key]
            del obj[key]
    return obj


def dear_mongo():
    sessions = get_sessions()
    for session in sessions:
        print session
        for index, (root, dirs, files) in enumerate(os.walk(session)):
            # if index == 5: break
            if dirs == ["text-versions"]:
                if isfile(root + '/data.json'):
                    text_dict = {}
                    for version in [f for f in listdir(root + '/text-versions/') if isdir(root+'/text-versions/'+f)]:
                        doc_path = root + '/text-versions/' + version + '/document.txt'
                        if isfile(doc_path):
                            with open(doc_path) as f:
                                text = f.read()
                            text_dict.update({version: text})
                    with open(root + '/data.json') as j:
                        bill_data = json.load(j)
                    bill_data.update({'text_versions': text_dict})
                    # print("saving: {0}".format(str(root)))
                    bills.save(bill_data)
            if '/votes' in root:
                for file in files:
                    if file == 'data.json':
                        # print('saving: ' + root + '/' + file)
                        with open(root + '/' + file) as j:
                            j = json.load(j, object_hook=remove_dot_key)
                            try:
                                votes.save(j)
                            except:
                                print "except!"


dear_mongo()