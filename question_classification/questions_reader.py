
class QuestionsReader(object):

    def __init__(self):
        self.__questions = self.__read_questions

    def questions(self):
        return self.__questions

    def __read_questions(self):
        raise ValueError('You need to subclass this')
