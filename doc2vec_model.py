
from basic_model import BasicModel
from semantic_sentence_retrieval.doc2vec_sentence_retriever import Doc2VecSentenceRetriever


class Doc2VecSentenceRetrieverModel(BasicModel):
    def __init__(self, documents, train_data):
        BasicModel.__init__(self, documents, train_data)
        self.retreiver = Doc2VecSentenceRetriever(documents[0])