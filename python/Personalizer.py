import numpy as np
import platform
import geocoder
import Indexer
import WordCleaner
import Matcher

history_vectors = []


def get_country_name():
    g = geocoder.ip("me")
    country = str(g.country)
    city = str(g.city)
    if g.country is None:
        country = "syria"
        city = "damascus"
    country_city = country + " " + city
    return country_city.split()


def get_user_os():
    os = str(platform.system())
    return os.split()


def os_vector(model):
    os_list = get_user_os()
    os_vector_list = [model.wv[word] for word in os_list if word in model.wv]
    if os_vector_list:
        os_vector = np.mean(os_vector_list, axis=0).reshape(1, -1)
    return os_vector


def clear_history():
    history_vectors.clear()


def calculate_histories_vector():
    if len(history_vectors) > 0:
        # Create weights that give more importance to recent vectors
        weights = np.arange(1, len(history_vectors) + 1)
        # Calculate the weighted mean vector
        avg_histories_vector = np.average(history_vectors, axis=0, weights=weights)
        return avg_histories_vector


def get_ans_persona_word_2_vec(documents_vectors, query, dataset_keys, model):

    avg_histories_vector = calculate_histories_vector()

    processed_query = WordCleaner.query_cleaning(query)
    query_vector_list = [model.wv[word] for word in processed_query if word in model.wv]

    # Check if there are valid vectors to avoid nan issues
    if query_vector_list:

        query_vector = np.mean(query_vector_list, axis=0).reshape(1, -1)

        country_city = get_country_name()
        country_vector_list = [
            model.wv[word] for word in country_city if word in model.wv
        ]
        if country_vector_list:
            print(country_city)
            country_vector = np.mean(country_vector_list, axis=0).reshape(1, -1)
        else:
            country_vector = query_vector

        if len(history_vectors) == 0:
            print("no history")
            avg_histories_vector = query_vector
        else:
            print(len(history_vectors), "history queries")

        weighted_vector = (
            0.65 * query_vector + 0.25 * avg_histories_vector + 0.1 * country_vector
        )

        similar_docs = Matcher.get_query_answers(
            documents_vectors, weighted_vector, dataset_keys, 0.6
        )
        for i, (doc_id, score) in enumerate(list(similar_docs.items())[:10]):
            print(f"Rank {i+1}, Document ID: {doc_id}, Similarity Score: {score}")

        history_vectors.append(weighted_vector)
    else:
        print("None of the query words were found in the model's vocabulary.")


def get_ans_persona_transformers(query, model, documents_vectors, dataset_keys):

    avg_histories_vector = calculate_histories_vector()

    processed_query = WordCleaner.query_cleaning(query)

    embeddingsQ = model.encode(processed_query).reshape(1, -1)

    if len(history_vectors) == 0:
        print("no history")
        avg_histories_vector = embeddingsQ
    else:
        print(len(history_vectors), "history queries")

    country_city = get_country_name()
    print(country_city)
    country_vector = model.encode(country_city).reshape(1, -1)

    weighted_vector = (
        0.65 * embeddingsQ + 0.25 * avg_histories_vector + 0.1 * country_vector
    )
    similar_docs = Matcher.get_query_answers(
        documents_vectors, weighted_vector, dataset_keys, 0.55
    )
    for i, (doc_id, score) in enumerate(list(similar_docs.items())[:10]):
        print(f"Rank {i+1}, Document ID: {doc_id}, Similarity Score: {score}")

    history_vectors.append(weighted_vector)


def get_query_answers_personalized(
    documents_vectors,
    query,
    dataset_keys,
    model,
    threshold,
    dataset_name,
    personalize=True,
):
    if dataset_name == "wikir":
        query_vector_list = [model.wv[word] for word in query if word in model.wv]
        country_vector_list = [
            model.wv[word] for word in country_city if word in model.wv
        ]
        if query_vector_list:
            query_vector = np.mean(query_vector_list, axis=0).reshape(1, -1)
        else:
            print("None of the query words were found in the model's vocabulary.")
    else:
        query_vector = model.encode(query).reshape(1, -1)
    country_vector = query_vector
    avg_histories_vector = query_vector

    if personalize:
        if len(history_vectors) > 0:
            avg_histories_vector = calculate_histories_vector()
        country_city = get_country_name()

        if dataset_name == "wikir":
            if country_vector_list:
                country_vector = np.mean(country_vector_list, axis=0).reshape(1, -1)
        else:
            country_vector = model.encode(" ".join(country_city)).reshape(1, -1)
    weighted_vector = (
        0.65 * query_vector + 0.25 * avg_histories_vector + 0.1 * country_vector
    )
    if personalize:
        history_vectors.append(weighted_vector)
    similar_docs = Matcher.get_query_answers(
        documents_vectors, weighted_vector, dataset_keys, threshold
    )
    return similar_docs
