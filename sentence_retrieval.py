from __future__ import print_function
import json
import nltk

from math import log
from collections import defaultdict, Counter

stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

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


class BasicSentenceRetriever:
    def __init__(self, documents):
        self.documents = documents
        self._build_index()

    def _build_index(self):
        doc_term_freqs = {}
        for sentence in self.documents:
            term_freqs = extract_term_freqs(sentence)
            doc_term_freqs[self.documents.index(sentence)] = term_freqs
        M = len(doc_term_freqs)
        doc_freqs = compute_doc_freqs(doc_term_freqs)
        self.invertedIndex = defaultdict(list)
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
                    self.invertedIndex[term].append([docid, tfidf / length])

    def lookup(self, user_query):
        query = []
        for token in nltk.word_tokenize(user_query):
            if token not in stopwords:
                query.append(stemmer.stem(token.lower()))
        result = query_vsm(query, self.invertedIndex)
        try:
            return self.documents[result[0][0]]
        except:
            print('**************')
            print('ERROR')
            print('result:')
            print(result)
            print('len(self.documents):')
            print(len(self.documents))
            print('**************')
            return self.documents[0]