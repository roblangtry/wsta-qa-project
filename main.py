import json
from basic_model import BasicModel
DEV_FILE = 'data/QA_dev.json'
TEST_FILE = 'data/QA_test.json'
TRAIN_FILE = 'data/QA_train.json'


def main():
    dev_data = load_json_file(DEV_FILE)
    test_data = load_json_file(TEST_FILE)
    train_data = load_json_file(TRAIN_FILE)
    #model = BasicModel(get_documents(dev_data), get_answers_and_queries(train_data))
    # now run a simple test
    correct = 0
    total = 0
    n = len(dev_data)
    m = 1
    for obj in dev_data:
        print m,
        print '/',
        print n
        m += 1
        model = BasicModel([obj['sentences']], [])
        for o2 in obj['qa']:
            query = o2['question']
            answer = o2['answer']
            model_answer = model.answer_query(query)
            total += 1
            if model_answer == answer:
                correct += 1
    print 'Precision on dev data: ',
    print '%.2f' % (float(correct) / float(total) * float(100))


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