import streamlit as st
import Interface
import FileManager
import Matcher
import WordCleaner
import Indexer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dataset = st.selectbox(
    "Dataset",
    ["wikir", "lotte"],
    help="Choose the dataset to search",
    disabled=st.session_state.disabled,
    label_visibility=st.session_state.visibility,
)

if dataset:
    text_input = ""
    if dataset == "wikir":
        vectorizer, dataset_keys, sparse_matrix, matrix = (
            FileManager.load_model_from_drive("model/wikir")
        )
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectorizer, dataset_keys, sparse_matrix, matrix = (
            FileManager.load_model_from_drive("model/lotte")
        )

mode = st.selectbox(
    "Indexing Method",
    ["tf-idf", "embedding"],
    help="choose whether to use TF-IDF or Embedding",
    disabled=st.session_state.disabled,
    label_visibility=st.session_state.visibility,
)

if mode:
    text_input = ""


text_input = st.text_input(
    "Enter your query:",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

if text_input:
    query = WordCleaner.query_cleaning(text_input)
    if mode == "tf-idf":
        answers = Matcher.get_query_answers(
            sparse_matrix,
            Indexer.calculate_doc_tf_idf([" ".join(query)], vectorizer),
            dataset_keys,
            0.35,
        )
    else:
        answers = Matcher.get_query_answers(
            matrix,
            Indexer.calculate_doc_embedding(" ".join(query), model),
            dataset_keys,
            0.55,
        )
    st.write("You entered: ", text_input)
    col1, col2 = st.columns([4, 1])
    for answer in answers:
        text = Interface.get_row_by_id(dataset, answer)
        col1.container = st.container(border=True)
        col1.container.header(" ".join(word_tokenize(text[1])[0:3]))
        col1.container.write(text[1])
        col1.container.caption(":orange[ Result ID #{}]".format(answer))
