import streamlit as st
import gensim
from annoy import AnnoyIndex
import preprocess_annoy
import pickle
import random

model = preprocess_annoy.ANNOY_PATH / 'fasttext-wiki-news-subwords-300.ann'

with st.spinner("Loading Model..."):
    info = gensim.downloader.info()['models'][model.stem]
    u = AnnoyIndex(info['parameters']['dimension'], 'angular')
    u.load(str(model))

    with open(preprocess_annoy.ANNOY_PATH / f"{model.stem}.pkl", 'rb') as f:
        index_to_key = pickle.load(f)
        key_to_index = {v:i for i, v in enumerate(index_to_key)}


query = st.text_input(
    label="Query",
    value=st.query_params.get("q", "thesaurus")
)

if query:
    if query not in key_to_index:
        st.error("Word not found... try another")
    else:
        index = key_to_index[query]
        nns = u.get_nns_by_item(index, 1000, include_distances=True)
        for ranking, (nn, score) in enumerate(zip(*nns)):
            if ranking == 0:
                continue
            word = index_to_key[nn]
            st.write(f"#{ranking} [{(1 - score) * 100:.1f}%]: [{word}](/?q={word})")
