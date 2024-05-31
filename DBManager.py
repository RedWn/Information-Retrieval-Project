import pandas as pd
from sqlalchemy import create_engine

# pip install pandas sqlalchemy pymysql mysql-connector-python


def csv_to_mysql(csv_file_path):
    df = pd.read_csv(csv_file_path, delimiter="\t")
    engine = create_engine("mysql+pymysql://python@localhost/ir")
    df.to_sql("lotte", con=engine, if_exists="replace", index=False)


csv_to_mysql("lotte/collection.tsv")
