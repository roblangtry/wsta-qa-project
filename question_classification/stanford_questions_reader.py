from questions_reader import QuestionsReader
from entity_tagger import BasicEntityTagger
import json

class StanfordQuestionsReader(QuestionsReader):

#   Note: Should run on training data, not dev, but dev's smaller, so yeah...
    __train = 'data/QA_dev.json'
    # __train = 'data/QA_train.json'
    __test = 'data/QA_dev.json'

    def __init__(self):
        super(self.__class__, self).__init__()
        answer_documents = self.__get_answer_documents()
        self.__tagger = BasicEntityTagger(answer_documents)

    def __get_answer_documents(self):
        answer_documents = list()
        qanda_tuples = list()
        with open(self.__train) as f:
            json_data = json.load(f)
            for d in json_data:
                qanda = d['qa']
                for obj in qanda:
                    q = obj['question']
                    a = obj['answer']
                    qanda_tuples.append(tuple((q, a)))
                    answer_documents.append(a)
        self.__qanda_tuples = qanda_tuples
        return answer_documents

    def get_training_questions(self):
        return self.__get_questions()

    def get_test_questions(self):
        return []

    def __get_questions(self):
        questions = list()
        for t in self.__qanda_tuples:
            q = t[0]
            a = t[1]
#           This is a very slow operation
            tags = self.__tagger.tag(a)
            if len(tags) > 0:
                print " "
                for t in tags:
                    question = tuple((t[0], q))
                    questions.append(question)
                    print question
                print "------------------------------------------------------------"
        return questions
