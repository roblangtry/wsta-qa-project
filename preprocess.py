import nltk
import string
import json
punctuations = {';',':','!','?','/','\\','#','@','$','&',')','(','\"','>','<','-','_','+','=','{','}'}
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma

def normalize(sentence):
    words = []
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        if word not in punctuations:
                if word not in stopwords:
                    words.append(word)
    return words

def question(doc):
    questions = []
    for qa in doc['qa']:
        questions.append(normalize(qa['question']))
    return questions

def data_from_json(json_file):
    with open(json_file) as json_data:
        return json.load(json_data)
