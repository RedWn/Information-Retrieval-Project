import streamlit as st
import Interface
import FileManager
import Matcher
import WordCleaner
import Indexer
from nltk.tokenize import word_tokenize


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
    if dataset == "wikir":
        type = 0
        vectorizer, dataset_keys, matrix = FileManager.load_model_from_drive(
            "model/wikir", type
        )
    else:
        type = 1
        vectorizer, dataset_keys, matrix = FileManager.load_model_from_drive(
            "model/lotte", type
        )


text_input = st.text_input(
    "Enter your query:",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)
# "she co founded the phillips collection with her husband duncan phillips she was born marjorie acker in bourbon indiana she was the sister to six other siblings her parents were charles ernest acker and alice beal she was raised in ossining new york phillips started drawing as a child her uncles were reynolds beal and gifford beal both men noticed phillips artistic ability and suggested she pursue art as a career path she began attending the art students league in 1915 and graduated in 1918 she studied under boardman robinson marjorie phillips has the unmistakable style of the born painter duncan phillips phillips is quoted as stating that she didn t want to paint depressing pictures she painted primarily landscapes and still life works despite living a socialite lifestyle alongside her husband phillips made the effort to paint every morning in her washington d c studio she attended an art exhibition for duncan phillips at the century association in january 1921 she met duncan and the two married in october of that year duncan was an art collector and the couple expanded their collecting phillips moved to washington d c and into duncan s dupont circle mansion duncan s mother",

if text_input:
    query = WordCleaner.query_cleaning(text_input)
    if type == 0:
        answers = Matcher.get_query_answers(
            matrix,
            Indexer.calculate_doc_tf_idf([" ".join(query)], vectorizer),
            dataset_keys,
            0.5,
        )
    else:
        answers = Matcher.get_query_answers(
            matrix,
            Indexer.calculate_doc_embedding(" ".join(query)),
            dataset_keys,
            0.5,
        )
    st.write("You entered: ", text_input)
    col1, col2 = st.columns([4, 1])
    for answer in answers:
        text = Interface.get_row_by_id(dataset, answer)
        col1.container = st.container(border=True)
        col1.container.header(" ".join(word_tokenize(text[1])[0:3]))
        col1.container.write(text[1])
        col1.container.caption(":orange[ Result ID #{}]".format(answer))
