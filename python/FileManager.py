import csv
import pickle
import json
from nltk.tokenize import word_tokenize
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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


def tsv_to_dict(filename):
    with open(filename, mode="r") as infile:
        reader = csv.reader(infile, delimiter="\t")
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
                    "rank": 1,  # queries_answers[key][rowKey],
                    "score": queries_answers[key][rowKey],
                    "tag": 1,
                }
            )
    file.close()
    return


def write_model_to_drive(
    name, vectorizer: TfidfVectorizer, svd: TruncatedSVD, keys, matrix
):
    pickle_model(name, vectorizer)
    pickle_svd(name, svd)
    store_keys(name, keys)
    store_matrix(name, matrix)
    return


def load_model_from_drive(name) -> sparse:
    vectorizer = unpickle_model(name)
    svd = unpickle_svd(name)
    keys = load_keys(name)
    matrix = load_matrix(name)
    return vectorizer, svd, keys, matrix


def pickle_model(name: str, vectorizer: TfidfVectorizer):
    path = name + ".pickle"
    pickle.dump(vectorizer, open(path, "wb"))
    return path


def unpickle_model(name: str) -> TfidfVectorizer:
    path = name + ".pickle"
    vectorizer = pickle.load(open(path, "rb"))
    return vectorizer


def pickle_svd(name: str, svd: TruncatedSVD):
    path = name + "_svd.pickle"
    pickle.dump(svd, open(path, "wb"))
    return path


def unpickle_svd(name: str) -> TruncatedSVD:
    path = name + "_svd.pickle"
    svd = pickle.load(open(path, "rb"))
    return svd


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


def store_matrix(name: str, matrix: sparse):
    path = name + ".matrix"
    sparse.save_npz(path, matrix)
    return


def load_matrix(name: str):
    path = name + ".matrix.npz"
    return sparse.load_npz(path)


def jsonl_to_tsv(jsonl_file_path: str, tsv_file_path: str) -> None:
    try:
        with open(jsonl_file_path, "r") as jsonl_file, open(
            tsv_file_path, "w"
        ) as tsv_file:
            tsv_file.write("qid\tq0\tanswer_pid\tscore\n")  # Write header

            for line in jsonl_file:
                data = json.loads(line)
                qid = data.get("qid", "")
                score = 1  # data.get("score", "1")
                answer_pids = data.get("answer_pids", [])

                for answer_pid in answer_pids:
                    tsv_file.write(f"{qid}\t0\t{answer_pid}\t{score}\n")

        print(f"Conversion completed. TSV file saved at {tsv_file_path}")
    except Exception as e:
        print(f"Error converting JSONL to TSV: {e}")
