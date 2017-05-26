import preprocess as pre
import probabilistic_sentence_retrieval as model


def evaluate():
    data = pre.data_from_json('QA_train.json')
    totalQ = 0
    totalC = 0

    for sent in data:
        sentences = sent['sentences']
        num = 0
        correctA = 0
        model.best_match(sentences, k1=1.2, b=0.75)
        questions = pre.question(sent)
        for question in questions:
            result = model.query_best_match(question, k3=100)
            if result and result[0][0] == sent['qa'][num]['answer_sentence']:
                correctA += 1
            num += 1
        totalQ += num
        totalC += correctA
    return totalC/totalQ

print(evaluate())
