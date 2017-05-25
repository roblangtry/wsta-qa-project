from question_classifier import QuestionClassifier
import nltk, re
from nltk.corpus import wordnet as wn

class HuangQuestionClassifier(QuestionClassifier):

    __WH_EXP = re.compile(r'\bwhat\b | \bwhich\b | \bwhen\b | \bwhere\b | \bwho\b | \bhow\b | \bwhy\b', flags=re.I | re.X)

    def __init__(self, questions_reader):
        super(self.__class__, self).__init__(questions_reader)



    def head_word(self, wh, question):
        tokens = nltk.word_tokenizer(question)
        if wh in ['when', 'where', 'why']:
            return False
        if wh in ['how']:
            return tokens[0]
        # if wh in ['what']:



    def features(self, question):
        features = dict()

        # try:
        print "--------------------------------------"
        print question
#           1. Wh- words
        REST = 'rest'
        matches = self.__WH_EXP.findall(question)
        print matches
        if len(matches) == 0:
            features['wh'] = REST
            head_question = question
        else:
            features['wh'] = matches[0]
            head_question = question.split(matches[0])[1]
        print head_question
#           The plain old string
        # features['question'] = question

#           Tokens
        # tokens = [ t for t in nltk.word_tokenize(question.lower()) if t not in self.__STOPWORDS and not all(c in self.__PUNCTUATION for c in t)]
        tokens = [ t for t in nltk.word_tokenize(question.lower()) if not all(c in self.PUNCTUATION for c in t)]
        # features['tokens'] = " ".join(tokens)

#           Stems
        stems = [self.STEMMER.stem(t) for t in tokens]
        # features['stems'] = " ".join(stems)

#           bigrams
        features['bigrams'] = " ".join([ a + " " + b for a, b in nltk.bigrams(stems)])

#           trigrams
        # features['trigrams'] = " ".join([ a + " " + b for a, b, c in nltk.ngrams(stems, 3)])

        # trigrams=ngrams(token,3)
#           POS tags
        pos = nltk.pos_tag(tokens)
        features['pos'] = " ".join([p for w, p in pos])

#           WordNet synonyms and hypernyms
#           Get first synonym and hypernym for all nouns, verbs, adjectives, adverbs
        types = [ 'NN', 'JJ', 'JJR', 'JJS', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        syn = list()
        hyp = list()
        for w, p in pos:
            if p in types:
                syns = wn.synsets(w)
                if len(syns) > 0:
                    syn.append(syns[0].lemmas()[0].name())
                    for hyper in syns[0].hypernyms():
                        hyp.append(hyper.lemmas()[0].name())
        # features['syn'] = " ".join(syn)
        features['hyp'] = " ".join(hyp)
        # except:
        #     print "Error reading line: {}".format(question)
#       Work here is done
        return features
