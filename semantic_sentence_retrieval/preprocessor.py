import nltk, string, time

class Preprocessor:

    __STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    __PUNCTUATION = string.punctuation

    @staticmethod
    def preprocess_documents(data):
        t = time.time()
        documents = list()
        for document in data:
#           Organise the sentences
            document_sentences = list()
            for sentence in document:
                document_sentences.append(Preprocessor.preprocess_sentence(sentence))
            documents.append(document_sentences)
        t = time.time() - t
        print "Preprocessed {} documents in {:.2f} seconds.".format(len(data), t)
        return documents

    @staticmethod
    def tokenize(s):
        return [ t for t in nltk.word_tokenize(s) if not all(c in Preprocessor.__PUNCTUATION for c in t)]


    @staticmethod
    def preprocess_sentence(sentence):
    #   Tokenize / Remove Stopwords / Ignore punctuation
        # tokens = [ t for t in nltk.word_tokenize(sentence.lower()) if t not in Preprocessor.__STOPWORDS and not all(c in Preprocessor.__PUNCTUATION for c in t)]
        # tokens = [ t for t in nltk.word_tokenize(sentence.lower()) if not all(c in Preprocessor.__PUNCTUATION for c in t)]
        tokens = [ t for t in nltk.word_tokenize(sentence) if not all(c in Preprocessor.__PUNCTUATION for c in t)]
    #   Lemmatize?
        # tokens = Preprocessor.lemmatize(tokens)
        return tokens

    @staticmethod
    def lemmatize(tokens):
        stemmer = nltk.stem.PorterStemmer()
        return [stemmer.stem(t) for t in tokens]
