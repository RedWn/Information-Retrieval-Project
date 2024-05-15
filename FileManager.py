import csv
from nltk.tokenize import word_tokenize


# class FileWriter:
def openCSVRead(pathToFile):
    csvfile = open(pathToFile)
    return csv.DictReader(csvfile, delimiter=",")


def openCSVWriter(pathToFile, fieldnames, delimiter=","):
    csvfile = open(pathToFile, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    return writer, csvfile


def csvToDict(filename):
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
        path, ["query_id", "iteration", "doc_id", "rank", "score", "tag"], "\t"
    )
    for key in queries.keys():
        for rowKey in queries_answers[key].keys():
            if queries_answers[key][rowKey] > 0.9:
                value = 2
            else:
                value = 1
            file_writer.writerow(
                {
                    "query_id": key,
                    "iteration": "Q0",
                    "doc_id": rowKey,
                    "rank": value,
                    "score": value,
                    "tag": value,
                }
            )
    file.close()
    return
