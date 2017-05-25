from questions_reader import QuestionsReader
from nltk.tag import StanfordNERTagger
import json

class StanfordQuestionsReader(QuestionsReader):

    # __train = 'data/QA_train.json'
    __train = 'data/QA_dev.json'
    __test = 'data/QA_dev.json'

    def __init__(self):
        super(self.__class__, self).__init__()
        self.__tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

    def get_training_questions(self):
        return self.__get_questions(self.__train)

    def __get_questions(self, fp):
        questions = list()
        with open(fp) as f:
            json_data = json.load(f)
            for d in json_data:
                qanda = d['qa']
                for obj in qanda:
                    # try:
                    coarse = self.__tagger.tag(obj['answer'])
                    question = obj['question']
                    questions.append(tuple((coarse, question)))
                    # except:
                        # print "Unable to open a line in {}.".format(fp)
                break
        return questions
