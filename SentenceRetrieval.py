from __future__ import print_function
import json
import nltk
from math import log

from math import log
from collections import defaultdict, Counter

stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))
TRAIN_FILE = 'data/QA_train.json'

with open(TRAIN_FILE) as data:
    data = json.load(data)

def extract_term_freqs(doc):
    tfs = Counter()
    for token in nltk.word_tokenize(doc):
        if token not in stopwords:
            tfs[stemmer.stem(token.lower())] += 1
    return tfs

def compute_doc_freqs(doc_term_freqs):
    dfs = Counter()
    for tfs in doc_term_freqs.values():
        for term in tfs.keys():
            dfs[term] += 1
    return dfs


def query_vsm(query, index, k=10):
    accumulator = Counter()
    for term in query:
        postings = index[term]
        for docid, weight in postings:
            accumulator[docid] += weight
    return accumulator.most_common(k)



i = 0
cSentence = []
for sentences in data:
    i+= 1
    doc = sentences['sentences']
    doc_term_freqs = {}
    for sent in doc:
        term_freqs = extract_term_freqs(sent)
        doc_term_freqs[doc.index(sent)] = term_freqs
    M = len(doc_term_freqs)

    doc_freqs = compute_doc_freqs(doc_term_freqs)

    invertedIndex = defaultdict(list)
    for docid, term_freqs in doc_term_freqs.items():
        N = sum(term_freqs.values())
        length = 0

        # TF-IDF values
        tfidf_values = []
        for term, count in term_freqs.items():
            tfidf = float(count) / N * log(M / float(doc_freqs[term]))
            tfidf_values.append((term, tfidf))
            length += tfidf ** 2

        # Normalize documents by length and insert them into index
        length = length ** 0.5
        for term, tfidf in tfidf_values:
            # note the inversion of the indexing, to be term -> (doc_id, score)
            if length != 0:
                invertedIndex[term].append([docid, tfidf / length])

    a = []
    totQuestions = 0
    correctOP = 0
    for qa in sentences['qa']:
        query = ""
        for token in nltk.word_tokenize(qa['question']):
            if token not in stopwords:
                query = query + ' ' + token
        result = query_vsm([stemmer.stem(term.lower()) for term in query.split()], invertedIndex)
        totQuestions += 1
        if len(result) > 0:
            bestSentence = result[0][0]
            if qa['answer_sentence'] == bestSentence:
                correctOP += 1
        cSentence.append((query,sentences['sentences'][result[0][0]]))

print("The accuracy on train set is", (correctOP/float(totQuestions)))

#The list with the questions and answers in a tuple form
#cSentence is the list which contains answers along with the answers in the form, (Question,Answer)
print(len(cSentence))
for i in cSentence:
    print((i)) 
