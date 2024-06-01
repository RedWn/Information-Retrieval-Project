from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer


def calculate_doc_tf_idf(query, vectorizer: TfidfVectorizer):
    return vectorizer.transform(query)


def calculate_doc_embedding(query):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(query)


def calculate_tf_idf(corpus, vectorizer: TfidfVectorizer):
    string_corpus = [str(element) for element in corpus.values()]
    return vectorizer.fit_transform(string_corpus)
