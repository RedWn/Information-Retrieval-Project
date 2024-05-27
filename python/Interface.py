import FileManager
import WordCleaner
import Indexer
import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


def search(dataset_name: str, query: str):
    query = word_tokenize(query)
    query = WordCleaner.remove_stop_words(query)
    query = WordCleaner.lemmatize(query)
    vectorizer, dataset_keys, tfidf_matrix = FileManager.load_model_from_drive(
        "../" + dataset_name
    )
    matrix = Indexer.calculate_doc_tf_idf([" ".join(query)], vectorizer)
    similar_rows = Matcher.get_query_answers(tfidf_matrix, matrix, dataset_keys, 0.35)
    return similar_rows
