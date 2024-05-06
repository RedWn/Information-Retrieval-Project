from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calcCosSimWithCorpus(tfidf_dataframe, vector):
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
