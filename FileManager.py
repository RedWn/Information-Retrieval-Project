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
