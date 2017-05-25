import math
import nltk
from collections import defaultdict, Counter
import time
import string
import json
from basic_model import BasicModel

stemmer = nltk.stem.PorterStemmer()

stopwords = set(nltk.corpus.stopwords.words('english'))


punctuations = set(string.punctuation)
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

#Lemmatization
def lemmatize(word):
  lemma = lemmatizer.lemmatize(word, 'v')
  if lemma == word:
    lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


#Preprocessing the words in the sentences
def process_sen(sentence, remove_stopwords):
    words = []
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        if word not in punctuations:
            if not remove_stopwords:
                if word not in stopwords:
                    words.append(word)
            else:
                words.append(word)
    return words

# Collect term frequencies for each sentence in document
def extract_term_freqs(sen):
    # bag-of-words representation
    tfs = Counter()
    sentence = process_sen(sen, True)
    for token in sentence:
        tfs[lemmatize(token.lower())] += 1
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
        self.tune1 = float(1.2)
        self.tune2 = float(2.0)
        self.b = float(0.75)
        self.index_prob = defaultdict(list)
        self.best_matching(documents, self.tune1, self.b)


    #The best matching algorithm
    def best_matching(self, doc, tune1,b):
        doc_term_freqs = {}
        a_id = 0
        length_sum = 0
        for sentence in doc:
            term_freqs = extract_term_freqs(sentence)
            doc_term_freqs[a_id] = term_freqs
            a_id += 1
            length_sum += sum(term_freqs.values())

        M = len(doc_term_freqs)
        length_average = float(length_sum) / M
        doc_freqs = compute_doc_freqs(doc_term_freqs)

        # best match VSM
        for sent_id, term_freqs in doc_term_freqs.items():
            N = sum(term_freqs.values())
            tfidf = []
            for term, count in term_freqs.items():
                #Formula for bm25 tf and idf
                idf = math.log((M - float(doc_freqs[term]) + 0.5) / (float(doc_freqs[term]) + 0.5))
                K = tune1 * (1 - b + b * (float(N) / length_average)) + count
                tf_doc_len = float(tune1 + 1) * count / K
                score = idf * tf_doc_len
                tfidf.append((term, score))

            for term, score in tfidf:
                self.index_prob[term].append([sent_id, score])
        for term, sent_ids in self.index_prob.items():
            sent_ids.sort()

    def query_best_match(self, query, tune2, k=1):
        counts = Counter()
        for token in query:
            counts[lemmatize(token.lower())] += 1
        accumulator = Counter()
        for i in query:
            i = lemmatize(i.lower())
            qTF = float((tune2 + 1) * counts[i]) / (tune2 + counts[i])
            postings = self.index_prob[i]
            for sent_id, weight in postings:
                accumulator[sent_id] += weight * qTF
        acc = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
        return acc

    def lookup(self, user_query, backoff=0):
        query = []
        for token in nltk.word_tokenize(user_query):
            if token not in stopwords:
                query.append(stemmer.stem(token.lower()))
        result = self.query_best_match(query, self.tune2)
        if len(result) > backoff:
            return self.documents[result[backoff][0]]
        return self.documents[backoff - len(result)]


class ProbabilisticSentenceRetrieverModel(BasicModel):
    def __init__(self, documents, train_data):
        BasicModel.__init__(self, documents, train_data)
        self.retreiver = ProbabilisticSentenceRetriever(documents[0])
