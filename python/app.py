import streamlit as st
import FileManager
import Matcher
import WordCleaner
import Indexer
import Personalizer
import pandas as pd
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="wide")
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
    st.session_state["searched"] = False
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
    st.session_state["indexer"] = Indexer.Indexer(
        st.session_state["vectorizer"], st.session_state["model"]
    )
if "history_table" not in st.session_state:
    st.session_state["history_table"] = pd.DataFrame(columns=["Previous Queries"])
    st.session_state["personalization"] = False


def select_dataset():
    st.session_state["query"] = None
    if st.session_state["dataset"] == "wikir":
        model_string = "model/lotte"
        st.session_state["model"] = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        model_string = "model/wikir"
        (st.session_state["model"],) = FileManager.load_word2vec_model(
            "model/wikir/embedding_8_epoch_20.model"
        )
    (
        st.session_state["vectorizer"],
        st.session_state["dataset_keys"],
        st.session_state["sparse_matrix"],
        st.session_state["matrix"],
    ) = FileManager.load_model_from_drive(model_string)
    st.session_state["indexer"] = Indexer.Indexer(
        st.session_state["vectorizer"], st.session_state["model"]
    )


def select_mode():
    st.session_state["query"] = None


def search():
    st.session_state.searched = True


def clear_history():
    Personalizer.clear_history()
    st.session_state["history_table"] = pd.DataFrame(columns=["Previous Queries"])


col1, col2 = st.columns([0.25, 0.75])

with col2:
    st.session_state["dataset"] = st.selectbox(
        "Dataset",
        ["wikir", "lotte"],
        help="Choose the dataset to search",
        label_visibility=st.session_state.visibility,
        on_change=select_dataset,
    )

    st.session_state["mode"] = st.selectbox(
        "Indexing Method",
        ["tf-idf", "embedding"],
        help="choose whether to use TF-IDF or Embedding",
        label_visibility=st.session_state.visibility,
        on_change=select_mode,
    )

    st.session_state["query"] = st.text_input(
        "Enter your query:",
        label_visibility=st.session_state.visibility,
        on_change=search,
    )

    if st.session_state.searched and st.session_state["query"]:
        query = WordCleaner.query_cleaning(st.session_state["query"])
        if st.session_state["mode"] == "tf-idf":
            answers = Matcher.get_query_answers(
                st.session_state["sparse_matrix"],
                st.session_state["indexer"].calculate_doc_tf_idf([" ".join(query)]),
                st.session_state["dataset_keys"],
                0.35,
            )
        else:
            if st.session_state["query"] == "wikir":
                answers = Matcher.get_query_answers(
                    st.session_state["matrix"],
                    st.session_state["indexer"].calculate_doc_vector(" ".join(query)),
                    st.session_state["dataset_keys"],
                    0.35,
                )
            else:
                answers = Matcher.get_query_answers(
                    st.session_state["matrix"],
                    st.session_state["indexer"].calculate_doc_embedding(
                        " ".join(query)
                    ),
                    st.session_state["dataset_keys"],
                    0.35,
                )
        st.write("You entered: ", st.session_state["query"])
        st.session_state["history_table"].loc[
            st.session_state["history_table"].size
        ] = st.session_state["query"]
        subcol1, subcol2 = st.columns([4, 1])
        if len(list(answers.keys())) > 0:
            text = FileManager.get_rows_by_ids(
                st.session_state["dataset"], list(answers.keys())[:10]
            )
        for i, answer in enumerate(list(answers.keys())[:10]):
            subcol1.container = st.container(border=True)
            subcol1.container.header(" ".join(word_tokenize(text[i][1])[0:3]))
            subcol1.container.write(text[i][1])
            subcol1.container.caption(":orange[ Result ID #{}]".format(answer))
        st.session_state.searched = False


with col1:
    st.checkbox("Enable Personalization", key="personalization")
    st.table(
        st.session_state["history_table"],
    )

    st.button(
        "Clear History",
        on_click=clear_history,
        disabled=not st.session_state.personalization,
    )
