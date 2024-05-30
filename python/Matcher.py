from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse


def get_query_answers(corpus_matrix, query_matrix,keys, threshold=0.25):
    # query_matrix_sparse = sparse.csr_matrix(query_matrix)

    similarity_matrix = cosine_similarity(corpus_matrix, query_matrix).reshape(-1)
    similar_rows_indices = np.where(similarity_matrix > threshold)[0]

    # Precompute max values for each row
    # max_values = similarity_matrix.max(axis=1)

    # Use a generator expression instead of a dictionary comprehension
    similar_rows = {keys[i]: similarity_matrix[i] for i in similar_rows_indices}
    # similar_rows = dict(similar_rows)
    similar_rows = {
        k: v
        for k, v in sorted(similar_rows.items(), key=lambda item: item[1], reverse=True)
    }
    return similar_rows
