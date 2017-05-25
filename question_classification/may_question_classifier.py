from question_classifier import QuestionClassifier
import nltk
from nltk.corpus import wordnet as wn

class MayQuestionClassifier(QuestionClassifier):

    def __init__(self, questions_reader):
        QuestionClassifier.__init__(self, questions_reader)

    def features(self, question):
        features = dict()

        try:
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
        except:
            print "Error reading line: {}".format(question)
#       Work here is done
        return features

