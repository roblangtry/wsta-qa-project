from time import sleep
import re
import nltk
from sklearn import linear_model
from entity_tagger import BasicEntityTagger
from sentence_retrieval import BasicSentenceRetriever

class BasicAnswerRanker(object):
    def __init__(self, documents, train_data):
        self.classifier = BasicQueryClassifier(documents, train_data)
        self.rank_no = 1
        self.pos_cache = {}

    def rank_list(self, entity_list, query):
        first_pass_result = self.first_pass(entity_list, query)
        second_pass_result = self.second_pass(first_pass_result, query)
        third_pass_result = self.third_pass(second_pass_result, query)
        return third_pass_result

    def first_pass(self, entity_list, query):
        high_ranked = []
        low_ranked = []
        for entry in entity_list:
            if content_words_appear_in_query(entry, query):
                low_ranked.append(entry)
            else:
                high_ranked.append(entry)
        return [high_ranked, low_ranked]

    def second_pass(self, first_pass_result, query):
        tag = self.classify(query)
        result = []
        for l in first_pass_result:
            high_ranked = []
            low_ranked = []
            for entry in l:
                for entity in entry[1]:
                    if entity[0] == tag:
                        high_ranked.append((entry[0], entity[1]))
                    else:
                        low_ranked.append((entry[0], entity[1]))
            result.append(high_ranked)
            result.append(low_ranked)
        return result


    def third_pass(self, second_pass_result, query):
        result = []
        for l in second_pass_result:
            ra = []
            for entry in l:
                rank = self.get_rank(entry, query)
                ra.append((rank, entry[1]))
            ra = sorted(ra, key=lambda x: x[0])
            ra = map(lambda x: x[1], ra)
            result.extend(ra)
        return result

    def classify(self, query):
        return self.classifier.classify(query)

    def get_rank(self, entry, query):
        CLOSED_CLASS_TAGS = [ #  pronouns, determiners, conjunctions, modals and prepositions.
            'CC',   #  Coordinating conjunction
            'DT',   #  Determiner
            'IN',   #  Preposition or subordinating conjunction
            'MD',   #  Modal
            'PRP',  #  Personal pronoun
            'PRP$', #  Possessive pronoun
            'WDT',  #  Wh-determiner
            'WP',   #  Wh-pronoun
            'WP$'   #  Possessive wh-pronoun
            ]
        try:
            closed_words = self.pos_cache[query]
        except KeyError:
            text = nltk.word_tokenize(query.lower())
            try:
                tagged = nltk.pos_tag(text)
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
                tagged = nltk.pos_tag(text)
            closed = []
            for tup in tagged:
                tag = tup[1]
                word = tup[0]
                if tag not in CLOSED_CLASS_TAGS:
                    closed.append(word)
            self.pos_cache[query] = closed
            closed_words = closed
        rank = 1000
        if len(closed_words) > 0:
            tags_to_check = self.check_tags(entry[0], closed_words)
            if len(tags_to_check) > 0:
                text = ' '.join(nltk.word_tokenize(entry[0].lower()))
                text = text.split(entry[1].lower())
                if(len(text) > 1):
                    for i in range(len(text) - 1):
                        lower = text[i].split()
                        low_val = -1
                        upper = text[i + 1].split()
                        up_val = -1
                        for i in range(len(lower)):
                            if lower[-1 - i] in tags_to_check:
                                low_val = i + 1
                        for i in range(len(upper)):
                            if upper[i] in tags_to_check:
                                up_val = i + 1
                        if low_val > 0 and low_val < rank:
                            rank = low_val
                        if up_val > 0 and up_val < rank:
                            rank = up_val
        return rank

    def check_tags(self, sentence, closed_words):
        lsentence = nltk.word_tokenize(sentence.lower())
        result = []
        for word in closed_words:
            if word in lsentence:
                result.append(word)
        return result


def content_words_appear_in_query(entry, query):
    content_words = map(lambda x: x[1], entry[1])
    for word in content_words:
        if word in query:
            return True
    return False

class BasicQueryClassifier:
    def __init__(self, documents, train_data):
        # Tags are: PERSON, LOCATION, NUMBER, OTHER
        self.documents = documents
        self.train_data = train_data
        self.yes = 0
        self.no = 0
        self.rules = [
            ('person', 'PERSON'),
            ('location', 'LOCATION'),
            ('number', 'NUMBER'),
            ('who','PERSON'),
            ('name', 'PERSON'),
            ('where', 'LOCATION'),
            ('located', 'LOCATION'),
            ('when was','NUMBER'),
            ('how many','NUMBER'),
            ('year','NUMBER'),
            ('decade','NUMBER'),
            ('percentage', 'NUMBER'),
            ('date', 'NUMBER'),
            ('when', 'NUMBER'),
            ('how much', 'NUMBER'),
            ('century', 'NUMBER'),
            ('what is the average', 'NUMBER'),
            ('countries', 'LOCATION'),
            ('country', 'LOCATION'),
            ('city', 'LOCATION'),
            ('time', 'NUMBER'),
            ('territory', 'LOCATION'),
            ('country', 'LOCATION'),
            ('countries', 'LOCATION'),
            ('how', 'OTHER'),
            ('term for', 'OTHER'),
            ('type of', 'OTHER'),
            ('population', 'NUMBER'),
            ('what group', 'OTHER'),
            ('what did', 'OTHER'),
            ('stand for', 'OTHER'),
            ('what kind of', 'OTHER'),
            ('an example of', 'OTHER'),
            ('what types of', 'OTHER'),
            ('the purpose of', 'OTHER'),
            ('used for?', 'OTHER'),
            ('what language', 'OTHER'),
            ('was the title', 'OTHER'),
        ]
    def classify(self, query):
        for rule in self.rules:
            word = rule[0]
            tag = rule[1]
            if word in query.lower():
                return tag
        return 'OTHER'


class LogisticRegressionRanker(object):
    def __init__(self, documents, train_data):
        self.classifier = BasicQueryClassifier(documents, train_data)
        self.rank_no = 1
        self.pos_cache = {}
        self.logreg = linear_model.LogisticRegression(C=1e5)
        self.train(train_data)

    def train(self, train_data):
        X = []
        Y = []
        tot = len(train_data)
        p = 1
        for obj in train_data[:10]:
            retreiver = BasicSentenceRetriever(obj['sentences'])
            tagger = BasicEntityTagger([obj['sentences']])
            for o2 in obj['qa']:
                query = o2['question']
                answer = o2['answer']
                sentences = [retreiver.lookup(query, backoff=0),retreiver.lookup(query, backoff=1),retreiver.lookup(query, backoff=2)]
                entity_list = []
                i = 0
                for sentence in sentences:
                    i += 1
                    entities = tagger.tag(sentence)
                    entity_list.append((sentence, entities, i))
                score_matrix = self.score_list(entity_list, query)
                X.extend(map(lambda x: (x[0], x[1], x[2], x[4]), score_matrix))
                Y.extend(map(lambda x: "Y" if x[3] == answer else "N", score_matrix))
        self.logreg.fit(X, Y)

    def score_list(self, entity_list, query):
        first_pass_result = self.first_pass(entity_list, query)
        second_pass_result = self.second_pass(first_pass_result, query)
        third_pass_result = self.third_pass(second_pass_result, query)
        return third_pass_result

    def rank_list(self, entity_list, query):
        score_matrix = self.score_list(entity_list, query)
        X = map(lambda x: (x[0], x[1], x[2], x[4]), score_matrix)
        preds = self.logreg.predict_proba(X)
        rank = map(lambda x: x[1], preds)
        ranked_results = []
        for i in range(len(rank)):
            ranked_results.append((rank[i], score_matrix[i][3]))
        ranked_results = sorted(ranked_results, key=lambda x: 1 - x[0])
        return map(lambda x: x[1], ranked_results)

    def first_pass(self, entity_list, query):
        result = []
        for entry in entity_list:
            if content_words_appear_in_query(entry, query):
                result.append((entry, 0))
            else:
                result.append((entry, 1))
        return result

    def second_pass(self, first_pass_result, query):
        tag = self.classify(query)
        result = []
        for entry in first_pass_result:
            for entity in entry[0][1]:
                if entity[0] == tag:
                    result.append((1, entry[1], entry[0][0], entity[1], entry[0][2]))
                else:
                    result.append((0, entry[1], entry[0][0], entity[1], entry[0][2]))
        return result


    def third_pass(self, second_pass_result, query):
        result = []
        for entry in second_pass_result:
            rank = self.get_rank(entry, query)
            result.append((rank, entry[0], entry[1], entry[3], entry[4]))
        return result

    def classify(self, query):
        return self.classifier.classify(query)

    def get_rank(self, entry, query):
        CLOSED_CLASS_TAGS = [ #  pronouns, determiners, conjunctions, modals and prepositions.
            'CC',   #  Coordinating conjunction
            'DT',   #  Determiner
            'IN',   #  Preposition or subordinating conjunction
            'MD',   #  Modal
            'PRP',  #  Personal pronoun
            'PRP$', #  Possessive pronoun
            'WDT',  #  Wh-determiner
            'WP',   #  Wh-pronoun
            'WP$'   #  Possessive wh-pronoun
            ]
        try:
            closed_words = self.pos_cache[query]
        except KeyError:
            text = nltk.word_tokenize(query.lower())
            try:
                tagged = nltk.pos_tag(text)
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
                tagged = nltk.pos_tag(text)
            closed = []
            for tup in tagged:
                tag = tup[1]
                word = tup[0]
                if tag not in CLOSED_CLASS_TAGS:
                    closed.append(word)
            self.pos_cache[query] = closed
            closed_words = closed
        rank = 1000
        if len(closed_words) > 0:
            tags_to_check = self.check_tags(entry[2], closed_words)
            if len(tags_to_check) > 0:
                text = ' '.join(nltk.word_tokenize(entry[2].lower()))
                text = text.split(entry[3].lower())
                if(len(text) > 1):
                    for i in range(len(text) - 1):
                        lower = text[i].split()
                        low_val = -1
                        upper = text[i + 1].split()
                        up_val = -1
                        for i in range(len(lower)):
                            if lower[-1 - i] in tags_to_check:
                                low_val = i + 1
                        for i in range(len(upper)):
                            if upper[i] in tags_to_check:
                                up_val = i + 1
                        if low_val > 0 and low_val < rank:
                            rank = low_val
                        if up_val > 0 and up_val < rank:
                            rank = up_val
        return rank

    def check_tags(self, sentence, closed_words):
        lsentence = nltk.word_tokenize(sentence.lower())
        result = []
        for word in closed_words:
            if word in lsentence:
                result.append(word)
        return result
