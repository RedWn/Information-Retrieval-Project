from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


def get_query_answers(corpus_matrix, query_matrix, keys, threshold=0.25):
    similarity_matrix = cosine_similarity(corpus_matrix, query_matrix)
    similar_rows_indices = np.where(similarity_matrix > threshold)[0]

    # Get the row names (index) from the DataFrame
    similar_rows = {keys[i]: similarity_matrix[i].max() for i in similar_rows_indices}
    return similar_rows
