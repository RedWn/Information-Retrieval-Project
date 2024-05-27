from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def calculate_doc_tf_idf(query, vectorizer: TfidfVectorizer):
    return vectorizer.transform(query)


def calculate_tf_idf(corpus, vectorizer: TfidfVectorizer):
    string_corpus = [str(element) for element in corpus.values()]
    return vectorizer.fit_transform(string_corpus)


def calculate_doc_lsa(query_matrix, svd: TruncatedSVD):
    return svd.transform(query_matrix)


def calculate_lsa(corpus_matrix, svd: TruncatedSVD):
    return svd.fit_transform(corpus_matrix)
