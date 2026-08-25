import streamlit as st
from search_engine import MiniSearchEngine

st.set_page_config(page_title="Mini Intelligent Search Engine", page_icon="🔎", layout="centered")

st.title("🔎 Mini Intelligent Search Engine")
st.write("Search a small knowledge base using TF-IDF, semantic similarity and lightweight query expansion.")

@st.cache_resource
def load_engine():
    return MiniSearchEngine()

engine = load_engine()

query = st.text_input("Search", placeholder="Try: machine learning, Python, databases, careers...")
top_k = st.slider("Number of results", 1, 8, 5)

if query:
    results = engine.search(query, top_k=top_k)
    st.subheader(f"Results for: {query}")
    if not results:
        st.info("No matching documents found.")
    else:
        for result in results:
            st.markdown(f"### {result['title']}")
            st.caption(f"Category: {result['category']}  •  Relevance: {result['score']:.3f}")
            st.write(result['content'])
            st.write("Keywords: " + ", ".join(result.get("keywords", [])))
            st.divider()
else:
    st.info("Enter a search query to see results.")
