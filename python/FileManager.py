import csv
import pickle
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import mysql.connector


def open_csv_writer(path_to_file, fieldnames, delimiter=",", headers=True):
    csvfile = open(path_to_file, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
    if headers:
        writer.writeheader()
    return writer, csvfile


def csv_to_dict(filename, delimiter=",", skip_headers=True):
    with open(filename, mode="r") as infile:
        reader = csv.reader(infile, delimiter=delimiter)
        if skip_headers:
            next(reader, None)  # skip the headers
        dict_from_csv = {rows[0]: word_tokenize(rows[1]) for rows in tqdm(reader)}
    return dict_from_csv


def write_dataset_to_file(path, corpus):
    file_writer, file = open_csv_writer(path, ["id", "text"])
    for key in corpus:
        file_writer.writerow({"id": key, "text": " ".join(corpus[key])})
    file.close()
    return


def write_runfile_to_file(path, queries, queries_answers, max_relevance=2):
    file_writer, file = open_csv_writer(
        path, ["query_id", "iteration", "doc_id", "rank", "score", "tag"], "\t", False
    )
    for key in tqdm(queries.keys()):
        for rowKey in queries_answers[key].keys():
            file_writer.writerow(
                {
                    "query_id": key,
                    "iteration": "Q0",
                    "doc_id": rowKey,
                    "rank": 1,
                    "score": max_relevance * queries_answers[key][rowKey],
                    "tag": 1,
                }
            )
    file.close()
    return


def write_model_to_drive(
    name,
    vectorizer: TfidfVectorizer,
    keys,
    sparse_matrix,
    matrix,
):
    pickle_model(name, vectorizer)
    store_keys(name, keys)
    store_sparse_matrix(name, sparse_matrix)
    store_matrix(name, matrix)
    return


def load_model_from_drive(name: str):
    vectorizer = unpickle_model(name)
    keys = load_keys(name)
    sparse_matrix = load_sparse_matrix(name)
    matrix = load_matrix(name)
    return vectorizer, keys, sparse_matrix, matrix


def pickle_model(name: str, vectorizer: TfidfVectorizer):
    path = name + ".pickle"
    pickle.dump(vectorizer, open(path, "wb"))
    return path


def unpickle_model(name: str) -> TfidfVectorizer:
    path = name + ".pickle"
    vectorizer = pickle.load(open(path, "rb"))
    return vectorizer


def store_keys(name: str, keys: list[str]) -> None:
    path = name + ".keys"
    with open(path, "w") as file:
        for key in keys:
            file.write(key + "\n")
    return


def load_keys(name: str) -> list[str]:
    path = name + ".keys"
    with open(path, "r") as file:
        keys = [line.strip() for line in file]
    return keys


def store_sparse_matrix(name: str, matrix: sparse):
    path = name + ".matrix"
    sparse.save_npz(path, matrix)
    return


def load_sparse_matrix(name: str):
    path = name + ".matrix.npz"
    return sparse.load_npz(path)


def store_matrix(name: str, matrix: np.array):
    path = name + ".npy"
    np.save(path, matrix)
    return


def load_matrix(name: str):
    path = name + ".npy"
    return np.load(path, allow_pickle=True)


def jsonl_to_tsv(jsonl_file_path: str, tsv_file_path: str) -> None:
    try:
        with open(jsonl_file_path, "r") as jsonl_file, open(
            tsv_file_path, "w"
        ) as tsv_file:
            tsv_file.write("qid\tq0\tanswer_pid\tscore\n")  # Write header

            for line in jsonl_file:
                data = json.loads(line)
                qid = "S" + str(data.get("qid", ""))
                score = 1  # data.get("score", "1")
                answer_pids = data.get("answer_pids", [])

                for answer_pid in answer_pids:
                    tsv_file.write(f"{qid}\t0\t{answer_pid}\t{score}\n")

        print(f"Conversion completed. TSV file saved at {tsv_file_path}")
    except Exception as e:
        print(f"Error converting JSONL to TSV: {e}")


def load_word2vec_model(model_path, npy_path):
    model = Word2Vec.load(model_path)
    documents_vectors = np.load(npy_path)
    return model, documents_vectors


def save_word2vec_model(model, model_path, documents_vectors, npy_path):
    model.save(model_path)
    np.save(npy_path, documents_vectors)


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
