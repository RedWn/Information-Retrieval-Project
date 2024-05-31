import FileManager
import WordCleaner
import Indexer
import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import mysql.connector
from gensim import corpora, models


def get_row_by_id(table_name, id):
    # Establish a database connection
    db_connection = mysql.connector.connect(
        host="localhost", user="python", database="ir"
    )

    # Create a new cursor
    cursor = db_connection.cursor()

    # Execute the query
    query = f"SELECT * FROM {table_name} WHERE id = %s"
    cursor.execute(query, (id,))

    # Fetch the result
    row = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    db_connection.close()

    return row


def getTopic(text):

    dictionary = corpora.Dictionary([word] for word in word_tokenize(text))
    corpus = [dictionary.doc2bow([word]) for word in word_tokenize(text)]
    ldamodel = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
    return ldamodel.print_topics(num_topics=1, num_words=5)[0][1]
