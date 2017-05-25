from basic_model import BasicModel

class MayQuestionClassifierModel(BasicModel):
    def __init__(self, documents, train_data, question_classifier):
        BasicModel.__init__(self, documents, train_data)
        self.ranker.classifier = question_classifier
