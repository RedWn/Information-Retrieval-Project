import ir_measures
from ir_measures import *
import pandas as pd


def evaluate(qrel_path, run_path, parameters):
    # TODO: convert [nDCG @ 10, P @ 5, P(rel=2) @ 5, Judged @ 10] to parameters
    qrels = ir_measures.read_trec_qrels(qrel_path)
    run = ir_measures.read_trec_run(run_path)

    ans = ir_measures.calc_aggregate(
        [nDCG @ 10, P @ 5, P(rel=2) @ 5, Judged @ 10], qrels, run
    )

    return ans
