import couchdb
import FileManager

couch = couchdb.Server("https://admin:admin@localhost:5984/_utils/#/_all_dbs")
# couch.login(name="red", password="123456789")
# db = couch["infrared-search"]
db = couch.create("test")
# dataset = FileManager.csv_to_dict("../wikir/csv/wikir.csv")
