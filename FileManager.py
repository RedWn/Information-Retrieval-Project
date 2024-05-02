import csv


# class FileWriter:
def openCSVRead(pathToFile):
    csvfile = open(pathToFile)
    return csv.DictReader(csvfile, delimiter=",")


def openCSVWriter(pathToFile, fieldnames):
    csvfile = open(pathToFile, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def csvToDict(filename):
    with open(filename, mode="r") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        dict_from_csv = {rows[0]: rows[1] for rows in reader}
    return dict_from_csv
