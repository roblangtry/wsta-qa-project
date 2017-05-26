import math
from collections import defaultdict, Counter
import preprocess as pre
from basic_model import BasicModel

# Using default values for the tuning parameters of the best match algorithm.
k1 = 1.2
k2 = 100
b = 0.75
R = 0.0
qf = 1

# Collect term frequencies for each sentence in document
def extract_term_freqs(sen):
    # bag-of-words representation
    tfs = Counter()
    sentence = pre.normalize(sen)
    for token in sentence:
        tfs[pre.lemmatize(token.lower())] += 1
    return tfs


# Compute document frequencies for each term
def compute_doc_freqs(doc_term_freq):
    dfs = Counter()
    for tfs in doc_term_freq.values():
        for term in tfs.keys():
            dfs[term] += 1
    return dfs


# p is the number of documents containing the 'term'
# M is the total number of documents
# k1 and b are free parameters, usually chosen,
# in absence of an advanced optimization, as k1 in [1.2,2.0] and b = 0.75
# N is the length of the document under inspection in words.
# formula for calculating the score taken from https://en.wikipedia.org/wiki/Okapi_BM25

def calculate_score(M,p,N,avdl,count):
    x = math.log((M - float(p) + 0.5) / (float(p) + 0.5))
    y = (float(k1+1) * count)/(count + (k1 * ((1-b) + b * (float(N)/float(avdl)))))
    z = ((k2+1) * qf) / (k2 + qf)
    return x * y * z

class ProbabilisticSentenceRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(list)
        self.best_match(documents)
    
    #Populate the best match index 
    def best_match(self,doc):
        self.index.clear()
        sentence_id = 0
        doc_term_freqs = {}
        length_sum = 0
        for sentence in doc:
            term_freqs = extract_term_freqs(sentence)
            doc_term_freqs[sentence_id] = term_freqs
            sentence_id += 1
            length_sum += sum(term_freqs.values())
            
        # Calculate the total number of term_freqs in the document.
        M = len(doc_term_freqs)

        # Calculate the average.
        avdl = float(length_sum) / M
        doc_freqs = compute_doc_freqs(doc_term_freqs)
        
        for ids, term_freqs in doc_term_freqs.items():
            N = sum(term_freqs.values())
            scores = []
            for term, count in term_freqs.items():
                docfreq = doc_freqs[term]
                score = calculate_score(M,docfreq,N,avdl,count)
                scores.append((term, score))

            # note the inversion of the indexing, to be term -> (doc_id, score)
            for term, score in scores:
                self.index[term].append([ids, score])

        # ensure posting lists are in sorted order
        for term, ids in self.index.items():
            ids.sort()

    # Query the index to get an answer
    def query_best_match(self, query):
        qfs = Counter()
        for token in query:
            qfs[pre.lemmatize(token.lower())] += 1
        accumulator = Counter()
        for term in query:
            term = pre.lemmatize(term.lower())
            postings = self.index[term]
            for ids, weight in postings:
                accumulator[ids] += weight 
        accumulator_list = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
        return accumulator_list
    
    # Return the results
    def lookup(self, user_query, backoff=0):
        question = pre.normalize(user_query)
        result = self.query_best_match(question, k3=100)
        if len(result) > backoff:
            return self.documents[result[backoff][0]]
        return self.documents[backoff - len(result)]



class ProbabilisticSentenceRetrieverModel(BasicModel):
    def __init__(self, documents, train_data):
        BasicModel.__init__(self, documents, train_data)
        self.retreiver = ProbabilisticSentenceRetriever(documents[0])
