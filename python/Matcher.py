from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from python import WordCleaner


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



def get_word2vec_ans(query, model, documents_vectors, dataset_keys):

    query = WordCleaner.query_cleaning(query)
    print(query)

    valid_vectors = [model.wv[word] for word in query if word in model.wv]

    # Check if there are valid vectors to avoid nan issues
    if valid_vectors:
        query_vector = np.mean(valid_vectors, axis=0).reshape(1, -1)
        # Compute cosine similarity between query and document vectors
        similar_docs = get_query_answers(documents_vectors, query_vector, dataset_keys, 0.55)
        # Print the IDs of the top 5 most similar documents
        for i, (doc_id, score) in enumerate(list(similar_docs.items())[:10]):
            print(f"Rank {i+1}, Document ID: {doc_id}, Similarity Score: {score}")
    else:
        print("None of the query words were found in the model's vocabulary.")
