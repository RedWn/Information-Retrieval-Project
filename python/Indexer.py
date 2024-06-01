from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_doc_tf_idf(query, vectorizer: TfidfVectorizer):
    return vectorizer.transform(query)


def calculate_doc_embedding(query, model):
    return model.encode(query).reshape(1, -1)


def calculate_tf_idf(corpus, vectorizer: TfidfVectorizer):
    string_corpus = [str(element) for element in corpus.values()]
    return vectorizer.fit_transform(string_corpus)
