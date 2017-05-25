import math
from collections import defaultdict, Counter
import preprocess as pre
from basic_model import BasicModel
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






class ProbabilisticSentenceRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(list)
        self.best_match(documents, k1=1.2, b=0.75)

    #Populate the best match index 
    def best_match(self,doc,k1,b):
        self.index.clear()
        sentence_id = 0
        doc_term_freqs = {}
        sentence_len_sum = 0
        for sentence in doc:
            term_freqs = extract_term_freqs(sentence)
            doc_term_freqs[sentence_id] = term_freqs
            sentence_id += 1
            sentence_len_sum += sum(term_freqs.values())
        M = len(doc_term_freqs)
        avg_sentence_len = float(sentence_len_sum) / M
        doc_freqs = compute_doc_freqs(doc_term_freqs)
        
        for sent_id, term_freqs in doc_term_freqs.items():
            N = sum(term_freqs.values())
            scores = []
            for term, count in term_freqs.items():
                idf = math.log((M - float(doc_freqs[term]) + 0.5) / (float(doc_freqs[term]) + 0.5))
                K = k1 * (1 - b + b * (float(N) / avg_sentence_len)) + count
                tf_doc_len = float(k1 + 1) * count / K
                score = idf * tf_doc_len
                scores.append((term, score))
            for term, score in scores:
                self.index[term].append([sent_id, score])
        for term, sent_ids in self.index.items():
            sent_ids.sort()

    # Query the index to get an answer
    def query_best_match(self, query, k3):
        qfs = Counter()
        for token in query:
            qfs[pre.lemmatize(token.lower())] += 1
        accumulator = Counter()
        for term in query:
            term = pre.lemmatize(term.lower())
            qtf = float((k3 + 1) * qfs[term]) / (k3 + qfs[term])
            postings = self.index[term]
            for sent_id, weight in postings:
                accumulator[sent_id] += weight * qtf
        accumulator_list = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
        return accumulator_list

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