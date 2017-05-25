import nltk, string
from nltk.corpus import wordnet as wn
from nltk import MaxentClassifier, NaiveBayesClassifier, DecisionTreeClassifier

CONVERT_TAG = {
    'NUM': 'NUMBER',
    'HUM': 'PERSON',
    'DESC': 'OTHER',
    'ABBR': 'OTHER',
    'ENTY': 'OTHER',
    'LOC': 'LOCATION'
}
class QuestionClassifier(object):

    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    PUNCTUATION = string.punctuation
    STEMMER = nltk.stem.PorterStemmer()

    MAX_TRAINING_ITER = 5


    def __init__(self, questions_reader):
        self.__questions_reader = questions_reader
        self.train()
        self.test()


    def train(self):
        features = list()
        for ( label, question ) in self.__questions_reader.get_training_questions():
            features.append((self.features(question), label))
        self.__classifier = MaxentClassifier.train(features, max_iter=self.MAX_TRAINING_ITER)


    def classify(self, question):
        f = self.features(question)
        return CONVERT_TAG[self.__classifier.classify(f)]


    def coarse_prob_classify(self, question):
        f = self.features(s)
        return self.__classifier.prob_classify(f)


    def test(self):
        total = 0
        correct = 0
        for ( label, question ) in self.__questions_reader.get_test_questions():
            total += 1
            # try:
            l = self.classify(question)
            if l == label:
                correct += 1
                # print "{}. {} : {} : {}".format(i, label, l, question)
            # except:
            #     print "Error reading question: {}".format(question)
        print "Question classifications: {} out of {} correct ({}%%)".format(correct, total, (float(correct) / float(total) * float(100)))
