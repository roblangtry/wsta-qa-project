from questions_reader import QuestionsReader
import nltk
nltk.download('qc')
from nltk.corpus import qc

class LiQuestionsReader(QuestionsReader):

    __train = qc.abspath('train.txt')
    __test = qc.abspath('test.txt')

    def __init__(self):
        super(self.__class__, self).__init__()


    def get_training_questions(self):
        return self.__get_questions(self.__train)

    def get_test_questions(self):
        return self.__get_questions(self.__test)


    def __get_questions(self, fp):
        questions = list()
        with open(fp) as f:
            for line in f:
                # try:
                questions.append(self.__read_question(line))
                # except:
                    # print "Unable to open a line in {}.".format(fp)
        return questions


    def __read_question(self, line):
        [coarse, _] = line.split(" ")[0].split(":")
        question = " ".join(line.split(" ")[1:])
        return ( coarse, question )
