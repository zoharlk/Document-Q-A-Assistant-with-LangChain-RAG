# this is the third piece of our pipeline: 
# a web UI on top of ingest.py (builds the index) and rag.py (answers questions).
# this few lines turn our script into a clickable web app (for the demo).

import streamlit as st
from rag import answer_question

st.set_page_config(page_title="Q&A Assistant", page_icon="📄")
st.title("Chat with your documents")
st.write("Ask a question and get an answer grounded in your own files.")
question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Searching your documents..."):
        answer, sources = answer_question(question)

    st.subheader("Answer")
    st.write(answer)
    st.subheader("Sources used")
    for d in sources:
        st.caption("- " + d.metadata.get("source", "unknown"))
