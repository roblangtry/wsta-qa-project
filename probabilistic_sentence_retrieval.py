import math
import nltk
from collections import defaultdict, Counter
import time
import string
import json

start_time = time.time()
stemmer = nltk.stem.PorterStemmer()

stopwords = set(nltk.corpus.stopwords.words('english'))
with open("/Users/abhisheksirohi/Desktop/PROJECT/data/QA_train.json",encoding = "ISO-8859-1") as f:
  data = json.load(f)

punctuations = set(string.punctuation)
index_prob = defaultdict(list)
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

#The best matching algorithm
def best_matching(doc,tune1,b):
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
            index_prob[term].append([sent_id, score])
    for term, sent_ids in index_prob.items():
        sent_ids.sort()

def query_best_match(query, tune2, k=1):
    counts = Counter()
    for token in query:
        counts[lemmatize(token.lower())] += 1
    accumulator = Counter()
    for i in query:
        i = lemmatize(i.lower())
        qTF = float((tune2 + 1) * counts[i]) / (tune2 + counts[i])
        postings = index_prob[i]
        for sent_id, weight in postings:
            accumulator[sent_id] += weight * qTF
    acc = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
    return acc
