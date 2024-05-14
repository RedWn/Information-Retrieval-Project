import ir_measures
from ir_measures import *
import pandas as pd

qrels = ir_measures.read_trec_qrels("wikir/qrels")
run = ir_measures.read_trec_run("testrun")


df = pd.read_csv("testrun", delimiter="\t")
run2 = ir_measures.read_trec_run(df)

test = ir_measures.calc_aggregate(
    [nDCG @ 10, P @ 5, P(rel=2) @ 5, Judged @ 10], qrels, run
)

print(test)
