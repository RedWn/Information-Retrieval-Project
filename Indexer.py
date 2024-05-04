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


def calculateDocTF_IDF(corpus, doc):
    tf_idf = {}
    doc_tf = calculateTF(doc)
    inverted_index = getInvertedIndex(corpus)
    idf = calculateIDF(corpus, inverted_index)
    for term in doc:
        tf_idf[term] = doc_tf[term] * idf[term]
    return tf_idf


def calculateManualTF_IDF(corpus):
    tfidf_matrix = []
    all_tokens = []
    for key in corpus:
        for token in corpus[key]:
            if token not in all_tokens:
                all_tokens.append(token)

    for key in corpus:
        tfidf_matrix.append(calculateDocTF_IDF(corpus, corpus[key]))
        for token in all_tokens:
            if token not in tfidf_matrix[-1].keys():
                tfidf_matrix[-1][token] = 0

    df = pd.DataFrame(
        tfidf_matrix,
    )

    return tfidf_matrix, df


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
