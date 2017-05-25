from basic_model import BasicModel
from entity_tagger import BasicEntityTagger
from answer_ranker import BasicAnswerRanker
from sentence_retrieval import BasicSentenceRetriever
import re

class InjectableModel(BasicModel):

    def __init__(self, documents, qas, retreiver=False, tagger=False, ranker=False):
        print 'Building Model ... ',
        self.documents = documents

#       Sentence Retreiver
        print 'Initialising Sentence Retreiver ... ',
        if retreiver:
            self.retreiver = retriever
        else:
            self.retreiver = BasicSentenceRetriever(documents[0])

#       Tagger
        print 'Initialising Sentence Tagger ... ',
        if tagger:
            self.tagger = tagger
        else:
            self.tagger = BasicEntityTagger(documents)

#       Answer Ranker
        print 'Initialising Answer Ranker ... ',
        if ranker:
            self.ranker = ranker
        else:
            self.ranker = BasicAnswerRanker(documents, qas)

        print 'Done!'
