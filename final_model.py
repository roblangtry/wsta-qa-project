from entity_tagger import BasicEntityTagger
from answer_ranker import BasicAnswerRanker
from probabilistic_sentence_retrieval import ProbabilisticSentenceRetriever
import re
class FinalModel(object):

    def __init__(self, documents, trainedRanker):
        #print 'Building Model ... ',
        self.documents = documents
        #print 'Initialising Sentence Retreiver ... ',
        self.retreiver = ProbabilisticSentenceRetriever(documents[0])
        #print 'Initialising Sentence Tagger ... ',
        self.tagger = BasicEntityTagger(documents)
        #print 'Initialising Answer Ranker ... ',
        self.ranker = trainedRanker
        #print 'Done!'

    def sentence_retrieval(self, query, documents, backoff=0):
        # documents will be a list each element of that list will in turn be a list of sentences from a wikipedia article
        # query will be a string
        # function should return a single sentence for each wikipedia article
        sentences = [self.retreiver.lookup(query, backoff=backoff),self.retreiver.lookup(query, backoff=backoff+1),self.retreiver.lookup(query, backoff=backoff+2)]
        # TODO code this
        #assert(len(documents) == len(sentences))
        return sentences


    def entity_extraction(self, sentences):
        # sentences will be a list of sentences each from a unique wikipedia article
        # function should return a list of tuples the first element of the tuple is the input sentence and the second
        # element is a list of entities in that sentence

        entity_list = []
        i = 0
        for sentence in sentences:
            i += 1
            entities = self.tagger.tag(sentence)
            entity_list.append((sentence, entities, i))

        #assert(len(sentences) == len(entity_list))
        return entity_list

    def answer_ranking(self, query, entity_list):
        # query will be a string
        # entity list is a list of tuples the first element of the tuple is the input sentence and the second
        # element is a list of entities in that sentence
        # these entities are tuples where the first element is the tag and the second is the object
        # e.g. (u'LOCATION', u'United States')
        # function should return a list of answers in order of their ranking

        ranked_answers = self.ranker.rank_list(entity_list, query)

        return ranked_answers

    def select_answer(self, ranked_answers):
        # ranked_answers is a list of answers in order of their ranking
        # function should return the selected answer
        answer = ''
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        #                              #
        # The following is test code!! #
        #                              #
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        if len(ranked_answers) > 0:
            answer = clean_answer(ranked_answers[0])
        else:
            answer = 'Unknown'
        # TODO code this
        return answer

    def answer_query(self, query):
        backoff = 0
        self.entity_list = [('',[])]
        while len(self.entity_list[0][1]) == 0:
            self.sentences = self.sentence_retrieval(query, self.documents, backoff=backoff)
            self.entity_list = self.entity_extraction(self.sentences)
            backoff += 1
        self.ranked_answers = self.answer_ranking(query, self.entity_list)
        return self.select_answer(self.ranked_answers)

def clean_answer(answer):
    pass1 = answer.replace('"', '')
    return pass1.replace(',', '-COMMA-')