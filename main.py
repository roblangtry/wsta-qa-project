import json

DEV_FILE = 'data/QA_dev.json'
TEST_FILE = 'data/QA_test.json'
TRAIN_FILE = 'data/QA_train.json'


def main():
    dev_data = load_json_file(DEV_FILE)
    test_data = load_json_file(TEST_FILE)
    train_data = load_json_file(TRAIN_FILE)


def load_json_file(filename):
    # code written with reference to http://stackoverflow.com/questions/20199126/reading-json-from-a-file
    with open(filename) as file:
        json_data = json.load(file)
    return json_data


if __name__ == '__main__':
    main()