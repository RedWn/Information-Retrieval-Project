from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse


def calculate_cos_similarity(tfidf_dataframe, vector):
    # Convert vector A to a numpy array
    A_array = np.array(vector)
    related_docs = []
    for index, row in tfidf_dataframe.iterrows():
        B_array = np.array(row)  # Access the row data (vector B)
        similarity = cosine_similarity([A_array], [B_array])[0][0]
        print(f"Cosine Similarity between doc 1 and doc {index}: {similarity:.4f}")
        if similarity > 0.5:
            related_docs.append((index, similarity))

    # Sort related_docs by similarity (highest to lowest)
    related_docs.sort(key=lambda x: x[1], reverse=True)

    return related_docs


# def get_query_answers(corpus_matrix, query_matrix, keys, threshold=0.25):
#     similarity_matrix = cosine_similarity(corpus_matrix, query_matrix)
#     similar_rows_indices = np.where(similarity_matrix > threshold)[0]

#     # Get the row names (index) from the DataFrame
#     similar_rows = {keys[i]: similarity_matrix[i].max() for i in similar_rows_indices}
#     return similar_rows


def get_query_answers(corpus_matrix, query_matrix, keys, threshold=0.25):
    # Convert the matrices to sparse CSR format
    query_matrix_sparse = sparse.csr_matrix(query_matrix)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(corpus_matrix, query_matrix_sparse, dense_output=False)

    # Convert similarity_matrix to a dense format if necessary
    similarity_matrix_dense = similarity_matrix.toarray()

    similar_rows_indices = np.where(similarity_matrix_dense > threshold)[0]
    similar_rows = {keys[i]: similarity_matrix_dense[i].max() for i in similar_rows_indices}
    return similar_rows
