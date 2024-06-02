from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Indexer:
    def __init__(self, vectorizer: TfidfVectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def calculate_doc_tf_idf(self, query):
        return self.vectorizer.transform(query)

    def calculate_doc_embedding(self, query):
        return self.model.encode(query).reshape(1, -1)

    def calculate_doc_vector(self, query):
        valid_vectors = [self.model.wv[word] for word in query if word in self.model.wv]
        if valid_vectors:
            query_vector = np.mean(valid_vectors, axis=0).reshape(1, -1)
            return query_vector
        return np.ndarray()

    def calculate_tf_idf(self, corpus):
        string_corpus = [str(element) for element in corpus.values()]
        return self.vectorizer.fit_transform(string_corpus)
