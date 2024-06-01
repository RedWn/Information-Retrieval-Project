import ir_measures
from ir_measures import *
import pandas as pd


def evaluate(qrel_path, run_path, max_rel=1):
    qrels = ir_measures.read_trec_qrels(qrel_path)
    run = ir_measures.read_trec_run(run_path)
    if max_rel == 2:
        ans = ir_measures.calc_aggregate([RR, P @ 10, R @ 10, AP(rel=2)], qrels, run)
    else:
        ans = ir_measures.calc_aggregate([RR, P @ 10, R @ 10, AP], qrels, run)

    return ans
