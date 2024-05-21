import ir_measures
from ir_measures import *
import pandas as pd


def evaluate(qrel_path, run_path):
    qrels = ir_measures.read_trec_qrels(qrel_path)
    run = ir_measures.read_trec_run(run_path)

    ans = ir_measures.calc_aggregate([RR, P @ 10, R @ 10, AP], qrels, run)

    return ans
