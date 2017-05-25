# from semantic_sentence_retriever import SemanticSentenceRetriever
from sentence_retrieval import BasicSentenceRetriever
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json, time
import multiprocessing
from preprocessor import Preprocessor

# class Doc2VecSentenceRetriever(SemanticSentenceRetriever):
class Doc2VecSentenceRetriever(BasicSentenceRetriever):

    def __init__(self, documents, qas):
        super(self.__class__, self).__init__(documents, qas)

    def _build_index(self):
        tagged_documents = list()
        self.documents_dict = dict()

        for i, s in enumerate(self.documents[0]):
            tokens = self.preprocess(s)
            tag = self.tag(i)
            d = TaggedDocument(tokens, [tag])
            tagged_documents.append(d)
            self.documents_dict[tag] = json.dumps(s)
        self.train(tagged_documents)


    def preprocess(self, s):
        return Preprocessor.tokenize(s)


    def train(self, documents):
        model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
        model.build_vocab(documents)
        for epoch in range(10):
            model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        self.__model = model


    def tag(self, i):
        return 'TAG_%s' % i


    def lookup(query):
        t = Preprocessor.preprocess_sentence(query)
        v = self.__model.infer_vector(t)
        return self.__model.docvecs.most_similar([v])
