# from semantic_sentence_retriever import SemanticSentenceRetriever
from sentence_retrieval import BasicSentenceRetriever
import json, time
from preprocessor import Preprocessor
import short_sentence_similarity

class LiSentenceRetriever(BasicSentenceRetriever):

    def __init__(self, documents):
        super(self.__class__, self).__init__(documents)

    def _build_index(self):
        self.__preprocessed_documents = list()
        for d in self.documents[0]:
            self.__preprocessed_documents.append(self.preprocess(d))

    def preprocess(self, s):
        return Preprocessor.preprocess_sentence_and_stopwords(s)

    def lookup(self, query):
        t = time.time()

        print "================================================================"
        print "LiSentenceRetriever lookup: "
        print "Q: {}".format(query)

        q = self.preprocess(query)
        max_score = 0
        for i, pd in enumerate(self.__preprocessed_documents):
            similarity = self.sentence_similarity1(q, pd)
            if similarity > max_score:
                max_score = similarity
                closest_document = self.documents[0][i]
                print "\nCurrent closest document:"
                print closest_document
            similarity = self.sentence_similarity2(q, pd)
            # if similarity > max_score:
            #     max_score = similarity
            #     closest_document = self.documents[0][i]
        print "\n\nClosest document:"
        print closest_document
        print ""
        t = time.time()-t
        print "LiSentenceRetriever lookup finished in a snap at {:.2f} seconds".format(t)
        print " "
        return closest_document


    def sentence_similarity1(self, t1, t2):
        return short_sentence_similarity.tokens_similarity(t1, t2, True)

    def sentence_similarity2(self, t1, t2):
        return short_sentence_similarity.tokens_similarity(t1, t2, False)
