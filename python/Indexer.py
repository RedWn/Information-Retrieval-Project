import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def calculate_doc_tf_idf(query, vectorizer: TfidfVectorizer):
    tfidf_matrix = vectorizer.transform(query)
    return tfidf_matrix


def calculate_tf_idf(corpus, vectorizer: TfidfVectorizer):
    string_corpus = [str(element) for element in corpus.values()]
    tfidf_matrix = vectorizer.fit_transform(string_corpus)
    return tfidf_matrix


def calculate_doc_lsa(query_matrix, svd: TruncatedSVD) -> TruncatedSVD:
    return svd.transform(query_matrix)


def calculate_lsa(corpus_matrix, svd: TruncatedSVD) -> TruncatedSVD:
    return svd.fit_transform(corpus_matrix)
