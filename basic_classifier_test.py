from answer_ranker import BasicQueryClassifier
import json
from collections import Counter
TRAIN_FILE = 'data/QA_train.json'
def get_answers_and_queries(data):
    qa = []
    for document in data:
        for query in document['qa']:
            qa.append((query['question'], query['answer']))
    return qa


def load_json_file(filename):
    # code written with reference to http://stackoverflow.com/questions/20199126/reading-json-from-a-file
    with open(filename) as file:
        json_data = json.load(file)
    return json_data

train_data = load_json_file(TRAIN_FILE)
classifier = BasicQueryClassifier([], [])
index = Counter()
index2 = Counter()
index3 = Counter()
index4 = Counter()
eindex = Counter()
eindex2 = Counter()
eindex3 = Counter()
eindex4 = Counter()
aq = get_answers_and_queries(train_data)
tagged = 0
total = 0
for q, a in aq:
    total += 1
    tag = classifier.classify(q)
    if tag == 'UNKNOWN':
        if(len(q.split()) >= 1):
            w = q.split()[-1]
            eindex[w.lower()] += 1
        if(len(q.split()) >= 2):
            w = q.split()[-2] + ' ' + w
            eindex2[w.lower()] += 1
        if(len(q.split()) >= 3):
            w = q.split()[-3] + ' ' + w
            eindex3[w.lower()] += 1
        if(len(q.split()) >= 4):
            w = q.split()[-4] + ' ' + w
            eindex4[w.lower()] += 1
        for w in q.split():
            index[w.lower()] += 1
        for i in range(len(q.split())-1):
            w = q.split()[i] + ' ' + q.split()[i+1]
            index2[w.lower()] += 1
        for i in range(len(q.split())-2):
            w = q.split()[i] + ' ' + q.split()[i+1] + ' ' + q.split()[i+2]
            index3[w.lower()] += 1
        for i in range(len(q.split())-3):
            w = q.split()[i] + ' ' + q.split()[i+1] + ' ' + q.split()[i+2] + ' ' + q.split()[i+3]
            index4[w.lower()] += 1
    else:
        tagged += 1
print 'Percent tagged ->',
print float(tagged) / float(total) * float(100),
print '%'
print 'UNIGRAM'
for w in index.most_common(80):
    print w
print 'BIGRAM'
for w in index2.most_common(60):
    print w
print 'TRIGRAM'
for w in index3.most_common(40):
    print w
print 'QUADGRAM'
for w in index4.most_common(40):
    print w
print 'endian UNIGRAM'
for w in eindex.most_common(20):
    print w
print 'endian BIGRAM'
for w in eindex2.most_common(20):
    print w
print 'endian TRIGRAM'
for w in eindex3.most_common(20):
    print w
print 'endian QUADGRAM'
for w in eindex4.most_common(20):
    print w