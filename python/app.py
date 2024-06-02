import streamlit as st
import Interface
import FileManager
import Matcher
import WordCleaner
import Indexer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

import random


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "dataset" not in st.session_state:
    st.session_state["dataset"] = "wikir"
if "model" not in st.session_state:
    st.session_state["model"] = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
if "query" not in st.session_state:
    st.session_state["query"] = ""
if (
    "vectorizer" not in st.session_state
    and "dataset_keys" not in st.session_state
    and "sparse_matrix" not in st.session_state
    and "matrix" not in st.session_state
):
    (
        st.session_state["vectorizer"],
        st.session_state["dataset_keys"],
        st.session_state["sparse_matrix"],
        st.session_state["matrix"],
    ) = FileManager.load_model_from_drive("model/wikir")


def select_dataset():
    st.session_state["query"] = None
    st.write(st.session_state["dataset"])
    if st.session_state["dataset"] == "wikir":
        model_string = "model/lotte"
    else:
        model_string = "model/wikir"
    st.session_state["model"] = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )  # TODO switch to word2vec
    (
        st.session_state["vectorizer"],
        st.session_state["dataset_keys"],
        st.session_state["sparse_matrix"],
        st.session_state["matrix"],
    ) = FileManager.load_model_from_drive(model_string)


def select_mode():
    st.session_state["query"] = None


st.session_state["dataset"] = st.selectbox(
    "Dataset",
    ["wikir", "lotte"],
    help="Choose the dataset to search",
    disabled=st.session_state.disabled,
    label_visibility=st.session_state.visibility,
    on_change=select_dataset,
)

st.session_state["mode"] = st.selectbox(
    "Indexing Method",
    ["tf-idf", "embedding"],
    help="choose whether to use TF-IDF or Embedding",
    disabled=st.session_state.disabled,
    label_visibility=st.session_state.visibility,
    on_change=select_mode,
)

st.session_state["query"] = st.text_input(
    "Enter your query:",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

if st.session_state["query"]:
    query = WordCleaner.query_cleaning(st.session_state["query"])
    if st.session_state["mode"] == "tf-idf":
        answers = Matcher.get_query_answers(
            st.session_state["sparse_matrix"],
            Indexer.calculate_doc_tf_idf(
                [" ".join(query)], st.session_state["vectorizer"]
            ),
            st.session_state["dataset_keys"],
            0.35,
        )
    else:
        answers = Matcher.get_query_answers(
            st.session_state["matrix"],
            Indexer.calculate_doc_embedding(" ".join(query), st.session_state["model"]),
            st.session_state["dataset_keys"],
            0.35,
        )
    st.write("You entered: ", st.session_state["query"])
    col1, col2 = st.columns([4, 1])
    if len(list(answers.keys())) > 0:
        text = Interface.get_rows_by_ids(
            st.session_state["dataset"], list(answers.keys())[:10]
        )
    for i, answer in enumerate(list(answers.keys())[:10]):
        col1.container = st.container(border=True)
        col1.container.header(" ".join(word_tokenize(text[i][1])[0:3]))
        col1.container.write(text[i][1])
        col1.container.caption(":orange[ Result ID #{}]".format(answer))
