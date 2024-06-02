from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def calculate_doc_tf_idf(query, vectorizer: TfidfVectorizer):
    return vectorizer.transform(query)


def calculate_doc_embedding(query, model):
    return model.encode(query).reshape(1, -1)


def calculate_doc_vector(query, model):
    valid_vectors = [model.wv[word] for word in query if word in model.wv]
    if valid_vectors:
        query_vector = np.mean(valid_vectors, axis=0).reshape(1, -1)
        return query_vector
    return np.ndarray()


def calculate_tf_idf(corpus, vectorizer: TfidfVectorizer):
    string_corpus = [str(element) for element in corpus.values()]
    return vectorizer.fit_transform(string_corpus)
