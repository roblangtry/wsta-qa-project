## Author: Abhishek Sirohi
##
##

import gensim
import os
import collections
import random
import json
import nltk
import preprocess as pre

file = 'QA_train.json'
sent_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')

size = 300
window = 15
minimum = 1
threshold = 1e-5
neg_size = 5
epoch = 100
dm = 0 
worker = 1 

# Method to Normalize the text 
def normalize(text):
    norm_text = text.lower()
    
    # Replace breaks with spaces
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, '')
    return norm_text
# Open the json file
with open(file,encoding = "ISO-8859-1") as f:
    data = json.load(f)
    
# Helper method to prepare the test corpus
def makeTest(lst):
    l = []
    for i in lst:
        p = nltk.word_tokenize(pre.lemmatize(i))
        l.append(p)
    return l

#Preparing test corpus
lst = []
for i in data:
    for j in i['qa']:
        for k,v in j.items():
            if(k == 'question'):
                lst.append(v)
                
test = makeTest(lst)

#Preparing the sentences
sents = []
for i in data:
    for j in i['sentences']:
        sgmnt = sent_segmenter.tokenize(j)
        sents.append(sgmnt)

# Creating the training corpus from the sentences in the jsoon file
def train_corpus(sents, tokens_only=False):
    for k,i in enumerate(sents):
        if tokens_only:
            yield gensim.utils.simple_preprocess(' '.join(i))
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(' '.join(i)), [k])

train = list(train_corpus(sents))
print(len(train))
mod = gensim.models.Doc2Vec(size=size, window=window, min_count=minimum, sample=threshold, workers=worker, hs=0, dm=dm, negative=neg_size, dbow_words=1, dm_concat=1,iter=epoch)
# Building the vocabulary here
mod.build_vocab(train)



ranks = []
second_ranks = []
for doc_id in range(len(train)):
    inf_vec = mod.infer_vector(train[doc_id].words)
    sims = mod.docvecs.most_similar([inf_vec], topn=len(mod.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    second_ranks.append(sims[1])
    
# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test))
inf_vec = mod.infer_vector(test[doc_id])
sims = mod.docvecs.most_similar([inf_vec], topn=len(mod.docvecs))

# Now we'll check for every question in the training file
results = []
for i in range(0,len(test)):
    doc_id = i
    inf_vec = mod.infer_vector(test[doc_id])
    sims = mod.docvecs.most_similar([inf_vec], topn=len(mod.docvecs))
    for label, index in [('MOST', 0)]:
        results.append((' '.join(test[i]),' '.join(train[sims[index][0]].words)))

#Result is the list of tuple of all the questions and their respective predicted answers by this doc2vec model.
#(question,prediction)
print(results)
