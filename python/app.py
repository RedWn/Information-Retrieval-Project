import streamlit as st
import Interface
import FileManager
import couchdb


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dataset = st.selectbox(
    "Dataset",
    ["wikir_RML", "lotte"],
    help="Choose the dataset to search",
    disabled=st.session_state.disabled,
    label_visibility=st.session_state.visibility,
)

# if dataset:
#     st.write("dataset_dict")
#     dataset_dict = FileManager.csv_to_dict("..\wikir\csv\wikir.csv")
#     st.write(dataset_dict)

text_input = st.text_input(
    "Enter your query:",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)
# "she co founded the phillips collection with her husband duncan phillips she was born marjorie acker in bourbon indiana she was the sister to six other siblings her parents were charles ernest acker and alice beal she was raised in ossining new york phillips started drawing as a child her uncles were reynolds beal and gifford beal both men noticed phillips artistic ability and suggested she pursue art as a career path she began attending the art students league in 1915 and graduated in 1918 she studied under boardman robinson marjorie phillips has the unmistakable style of the born painter duncan phillips phillips is quoted as stating that she didn t want to paint depressing pictures she painted primarily landscapes and still life works despite living a socialite lifestyle alongside her husband phillips made the effort to paint every morning in her washington d c studio she attended an art exhibition for duncan phillips at the century association in january 1921 she met duncan and the two married in october of that year duncan was an art collector and the couple expanded their collecting phillips moved to washington d c and into duncan s dupont circle mansion duncan s mother",
answers = Interface.search(dataset, text_input)

if text_input:
    st.write("You entered: ", text_input)
    col1, col2 = st.columns([4, 1])
    for answer in answers:
        col1.container = st.container(border=True)
        col1.container.header("Title")
        col1.container.expander("see more ")
        col1.container.write(long_text)
        col1.container.caption(":orange[ Result ID #{}]".format(x))
