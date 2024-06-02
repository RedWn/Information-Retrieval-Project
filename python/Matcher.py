from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_query_answers(corpus_matrix, query_matrix, keys, threshold=0.5):
    similarity_matrix = cosine_similarity(corpus_matrix, query_matrix).reshape(-1)
    similar_rows_indices = np.where(similarity_matrix > threshold)[0]

    # Use a generator expression instead of a dictionary comprehension
    similar_rows = {keys[i]: similarity_matrix[i] for i in similar_rows_indices}
    # similar_rows = dict(similar_rows)
    similar_rows = {
        k: v
        for k, v in sorted(similar_rows.items(), key=lambda item: item[1], reverse=True)
    }
    return similar_rows
