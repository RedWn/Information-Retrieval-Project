import FileManager
import WordCleaner
import Indexer
import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import mysql.connector
from gensim import corpora, models


def get_rows_by_ids(table_name, ids):
    # Establish a database connection
    db_connection = mysql.connector.connect(
        host="localhost", user="python", database="ir"
    )

    # Create a new cursor
    cursor = db_connection.cursor()

    # Prepare the format string for the query
    format_strings = ",".join(["%s"] * len(ids))

    # Execute the query
    query = f"SELECT * FROM {table_name} WHERE id IN ({format_strings})"
    cursor.execute(query, tuple(ids))

    # Fetch all the results
    rows = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    db_connection.close()

    return rows


def getTopic(text):

    dictionary = corpora.Dictionary([word] for word in word_tokenize(text))
    corpus = [dictionary.doc2bow([word]) for word in word_tokenize(text)]
    ldamodel = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
    return ldamodel.print_topics(num_topics=1, num_words=5)[0][1]
