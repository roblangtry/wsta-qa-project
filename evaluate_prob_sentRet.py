import preprocess as pre
import probabilistic_sentence_retrieval as model


def evaluate_model():
    # Reading the json file
    data = pre.data_from_json('data/QA_train.json')
    totalQ = 0
    totalC = 0

# Reading the data from the json file
    for sent in data:
        # Read the sentences
        sentences = sent['sentences']
        num = 0
        correctA = 0
        model.best_match(sentences)
        questions = []
        # Read the  dictionary values with 'qa' in it.
        for qa in sent['qa']:
            # Search the qa tag and check for the 'question' in the dictionary
            questions.append(pre.normalize(qa['question']))
        for question in questions:
            # Finding the results according to the model
            result = model.query_best_match(question)
            # Checking if the results match the answers given in the training data set. 
            if result and result[0][0] == sent['qa'][num]['answer_sentence']:
                correctA += 1
            num += 1
        totalQ += num
        totalC += correctA
    return totalC/totalQ
# Check the accuracy of the model
print("Accuracy of the model is: ",evaluate_model())
