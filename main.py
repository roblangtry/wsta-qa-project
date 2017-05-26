import json
import sys
from basic_model import BasicModel, clean_answer
from final_model import FinalModel
from answer_ranker import LogisticRegressionRanker
from logreg_model import LogRegModel
from probabilistic_sentence_retrieval import ProbabilisticSentenceRetrieverModel
from doc2vec_model import Doc2VecSentenceRetrieverModel
from may_model import MayQuestionClassifierModel
from question_classification import MayQuestionClassifier, RothQuestionsReader
DEV_FILE = 'QA_dev.json'
TEST_FILE = 'QA_test.json'
TRAIN_FILE = 'QA_train.json'


def main():
    dev_data = load_json_file(DEV_FILE)
    test_data = load_json_file(TEST_FILE)
    train_data = load_json_file(TRAIN_FILE)
    #model = BasicModel(get_documents(dev_data), get_answers_and_queries(train_data))
    # now run a simple test
    ranker = LogisticRegressionRanker([], train_data)
    questions_reader = RothQuestionsReader()
    qclassifier = MayQuestionClassifier(questions_reader)
    EnhancedModel = Doc2VecSentenceRetrieverModel
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
        answer_file = open('answers.csv', 'w')
        answer_file.write('id,answer\n')
        n = len(test_data)
        m = 1
        for obj in test_data:
            print m,
            print '/',
            print n
            m += 1
            model = FinalModel([obj['sentences']], ranker)
            for o2 in obj['qa']:
                query = o2['question']
                query_id = o2['id']
                model_answer = model.answer_query(query)
                answer_file.write('%s,%s\n' % (str(query_id), model_answer.encode('utf8')))
    elif len(sys.argv) > 1 and sys.argv[1] == '-bt':
        answer_file = open('answers.csv', 'w')
        answer_file.write('id,answer\n')
        n = len(test_data)
        m = 1
        for obj in test_data:
            print m,
            print '/',
            print n
            m += 1
            model = BasicModel([obj['sentences']], train_data)
            for o2 in obj['qa']:
                query = o2['question']
                query_id = o2['id']
                model_answer = model.answer_query(query)
                answer_file.write('%s,%s\n' % (str(query_id), model_answer.encode('utf8')))

    elif len(sys.argv) > 1 and sys.argv[1] == '-e':
        correct = 0
        unknown = 0
        in_sent = 0
        close = 0
        total = 0
        n = len(dev_data)
        m = 1
        #ranker = LogisticRegressionRanker([], train_data)
        for obj in dev_data:
            print m,
            print '/',
            print n
            m += 1
            model = MayQuestionClassifierModel([obj['sentences']], train_data, qclassifier)
            #model = EnhancedModel([obj['sentences']], train_data)
            #model.ranker = ranker
            for o2 in obj['qa']:
                query = o2['question']
                answer = o2['answer']
                model_answer = model.answer_query(query)
                total += 1
                if model_answer == clean_answer(answer):
                    correct += 1
                if answer in model.ranked_answers:
                    close += 1
                if answer in model.sentences[0]:
                    in_sent += 1
        print 'Enhanced model performance'
        print '  Correct answer guessed: ',
        print '%.2f' % (float(correct) / float(total) * float(100))
        print '  Answer found in sentence: ',
        print '%.2f' % (float(close) / float(total) * float(100))
        print '  Answer in sentence: ',
        print '%.2f' % (float(in_sent) / float(total) * float(100))
    else:
        correct = 0
        unknown = 0
        in_sent = 0
        close = 0
        total = 0
        n = len(dev_data)
        m = 1
        for obj in dev_data:
            print m,
            print '/',
            print n
            m += 1
            model = BasicModel([obj['sentences']], train_data)
            for o2 in obj['qa']:
                query = o2['question']
                answer = o2['answer']
                model_answer = model.answer_query(query)
                total += 1
                if model_answer == clean_answer(answer):
                    correct += 1
                if answer in model.ranked_answers:
                    close += 1
                if answer in model.sentences[0]:
                    in_sent += 1
        print 'Baseline performance'
        print '  Correct answer guessed: ',
        print '%.2f' % (float(correct) / float(total) * float(100))
        print '  Answer found in sentence: ',
        print '%.2f' % (float(close) / float(total) * float(100))
        print '  Answer in sentence: ',
        print '%.2f' % (float(in_sent) / float(total) * float(100))



def load_json_file(filename):
    # code written with reference to http://stackoverflow.com/questions/20199126/reading-json-from-a-file
    with open(filename) as file:
        json_data = json.load(file)
    return json_data


def get_answers_and_queries(data):
    qa = []
    for document in data:
        for query in document['qa']:
            qa.append((query['question'], query['answer']))
    return qa


def get_documents(data):
    documents = []
    for document in data:
        documents.append(document['sentences'])
    return documents


if __name__ == '__main__':
    main()
