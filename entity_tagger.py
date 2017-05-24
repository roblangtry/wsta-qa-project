from nltk.tag import StanfordNERTagger
import nltk
import re

class BasicEntityTagger:
    def __init__(self, documents):
        self.tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
        self.cache = {}
        self.build_cache(documents)

    def build_cache(self, documents):
        all_sents = []
        for document in documents:
            all_sents.extend(document)
        split = map(lambda s: nltk.word_tokenize(s), all_sents)
        entities = self.tagger.tag_sents(split)
        parsed_entities = map(lambda e: self.parse_entities(e), entities)
        for i in range(len(all_sents)):
            self.cache[all_sents[i]] = parsed_entities[i]


    def parse_sentence(self, sentence):
        replaced = re.sub('[,.[\]();:?!]', ' ', sentence)
        replaced = re.sub('[^a-zA-Z0-9 -]', '', replaced)
        return replaced

    def split_sentence(self, sentence):
        return sentence.split()

    def parse_entities(self, entities):
        better_entities = []
        first = True
        for entity in entities:
            content = entity[0]
            tag = entity[1]
            if tag == 'O':
                if not first and len(content) > 0 and content[0].isupper():
                    tag = 'OTHER'
                # the following is written with reference too http://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
                elif any(char.isdigit() for char in content):
                    tag = 'NUMBER'
            if tag == 'ORGANIZATION':
                tag = 'OTHER'
            if first:
                first = False
            better_entities.append((tag, content))
        return contigous_tagging(better_entities)

    def tag(self, sentence):
        try:
            return self.cache[sentence]
        except KeyError:
            parsed_sentence = self.parse_sentence(sentence)
            words = self.split_sentence(parsed_sentence)
            entities = self.tagger.tag(words)
            parsed_entities = self.parse_entities(entities)
            self.cache[sentence] = parsed_entities
        return self.cache[sentence]

def contigous_tagging(in_list):
    tag = 'O'
    content = ''
    out_list = []
    for tup in in_list:
        if tag == tup[0]:
            content += ' '
            content += tup[1]
        else:
            if tag != 'O':
                out_list.append((tag, content))
            tag = tup[0]
            content = tup[1]
    if tag != 'O':
        out_list.append((tag, content))
    return out_list