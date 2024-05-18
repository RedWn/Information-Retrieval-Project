import streamlit as st
long_text = "Lorem ipsum. " * 10
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

text_input = st.text_input(
    "Search For some documents  🔍",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

if text_input:
    st.write("You entered: ", text_input)
    col1,col2 = st.columns([4,1])
    for x in range(10):
        col1.container = st.container(border=True)
        col1.container.header('Title')
        col1.container.expander('see more ')
        col1.container.write(long_text)
        col1.container.caption(':orange[ Result ID #{}]'.format(x))