import pandas as pd
import math
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def getInvertedIndex(corpus):
    inverted_index = defaultdict(list)
    for docId, doc in corpus.items():
        for term in doc:
            inverted_index[term].append(docId)
    return dict(inverted_index)


def calculateTF(doc: str):
    tf = {}
    doc_terms = word_tokenize(doc)
    for term in doc_terms:
        tf[term] = doc_terms.count(term) / len(doc_terms)
    # tf_df = pd.DataFrame(calculate_tf(corpus["doc_1"]), index=["tf"])
    return tf


def calculateIDF(corpus, inverted_index):
    idf = {}

    docs_count = len(corpus)

    for term, doc_ids in inverted_index.items():
        idf[term] = math.log((docs_count / len(doc_ids)) + 1)
    # idf_df = pd.DataFrame(calculate_idf(), index=["idf"])
    return idf


def calculateDocTF_IDF(corpus, doc):
    tf_idf = {}
    doc_terms = word_tokenize(doc)
    doc_tf = calculateTF(doc)
    inverted_index = getInvertedIndex(corpus)
    idf = calculateIDF(corpus, inverted_index)
    for term in doc_terms:
        tf_idf[term] = doc_tf[term] * idf[term]
    # tf_idf_df = pd.DataFrame(tf_idf, index=["tf_idf"])
    return tf_idf


def calculateTF_IDF(corpus):
    # documents = list(corpus.values())
    vectorizer = TfidfVectorizer()
    string_corpus = [str(element) for element in corpus.values()]
    tfidf_matrix = vectorizer.fit_transform(string_corpus)
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=corpus.keys(),
    )
    return tfidf_matrix, df
