import pandas as pd
from collections import defaultdict


def get_inverted_index(corpus):
    inverted_index = defaultdict(list)
    for docId, doc in corpus.items():
        for term in doc:
            if len(term) > 1:
                inverted_index[term].append(docId)
    return dict(inverted_index)


def calculate_doc_tf_idf(query, vectorizer):
    tfidf_matrix = vectorizer.transform(query)
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )
    return df


def calculate_tf_idf(corpus, vectorizer):
    string_corpus = [str(element) for element in corpus.values()]
    tfidf_matrix = vectorizer.fit_transform(string_corpus)
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=corpus.keys(),
    )
    return df
