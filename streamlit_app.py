import streamlit as st
import gensim.downloader

model = st.selectbox(
    label="Model",
    options=list(gensim.downloader.info()['models'].keys()),
    index=3,
)

query = st.text_input(
    label="Query"
)

vectors = gensim.downloader.load(model)
st.write(len(vectors.index_to_key))

if query:
    st.write(vectors.most_similar(query))


