import pandas as pd
import math
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np


def getInvertedIndex(corpus):
    inverted_index = defaultdict(list)
    for docId, doc in corpus.items():
        for term in doc:
            inverted_index[term].append(docId)
    return dict(inverted_index)


def calculateTF(doc: str):
    tf = {}
    # doc_terms = word_tokenize(doc)
    for term in doc:
        tf[term] = doc.count(term) / len(doc)
    # tf_df = pd.DataFrame(calculate_tf(corpus["doc_1"]), index=["tf"])
    return tf


def calculateIDF(corpus, inverted_index):
    idf = {}

    docs_count = len(corpus)

    for term, doc_ids in inverted_index.items():
        idf[term] = math.log((docs_count / len(doc_ids)) + 1)
    return idf


def calculateDocTF_IDF(corpus, all_tokens, doc):
    tf_idf = {}
    doc_tf = calculateTF(doc)
    inverted_index = getInvertedIndex(corpus)
    idf = calculateIDF(corpus, inverted_index)
    for term in doc:
        if term in idf.keys():
            tf_idf[term] = doc_tf[term] * idf[term]

    for token in all_tokens:
        if token not in tf_idf.keys():
            tf_idf[token] = 0
    df = pd.DataFrame(tf_idf, index=["tf_idf"])
    return tf_idf, df


def calculateManualTF_IDF(corpus):
    all_tokens = []

    for key in corpus:
        for token in corpus[key]:
            if token not in all_tokens:
                all_tokens.append(token)

    tfidf_matrix = []

    for key in corpus:
        tfidf_matrix.append(calculateDocTF_IDF(corpus, all_tokens, corpus[key])[0])

    df = pd.DataFrame(
        tfidf_matrix,
    )
    ans = csr_matrix(arg1=df, dtype=np.float64)
    return ans, df


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
