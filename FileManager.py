import csv
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy import sparse


def open_csv_writer(path_to_file, fieldnames, delimiter=",", headers=True):
    csvfile = open(path_to_file, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
    if headers:
        writer.writeheader()
    return writer, csvfile


def csv_to_dict(filename):
    with open(filename, mode="r") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        dict_from_csv = {rows[0]: word_tokenize(rows[1]) for rows in reader}
    return dict_from_csv


def write_dataset_to_file(path, corpus):
    file_writer, file = open_csv_writer(path, ["id", "text"])
    for key in corpus:
        file_writer.writerow({"id": key, "text": " ".join(corpus[key])})
    file.close()
    return


def write_runfile_to_file(path, queries, queries_answers):
    file_writer, file = open_csv_writer(
        path, ["query_id", "iteration", "doc_id", "rank", "score", "tag"], "\t", False
    )
    for key in queries.keys():
        for rowKey in queries_answers[key].keys():
            file_writer.writerow(
                {
                    "query_id": key,
                    "iteration": "Q0",
                    "doc_id": rowKey,
                    "rank": queries_answers[key][rowKey],
                    "score": queries_answers[key][rowKey],
                    "tag": queries_answers[key][rowKey],
                }
            )
    file.close()
    return


def write_model_to_file(path, matrix: sparse):
    sparse.save_npz(path, matrix)
    return


def load_model_from_file(path) -> sparse:
    return sparse.load_npz(path)
