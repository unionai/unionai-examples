"""Streamlit app for semantic quote search."""

# {{docs-fragment imports}}
import os
import random

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
# {{/docs-fragment imports}}

# Page configuration
st.set_page_config(
    page_title="Quote Search",
    page_icon="💬",
    layout="centered",
)

st.title("Quote Search")
st.write("Search for quotes by meaning using semantic similarity.")


# {{docs-fragment load-db}}
# Load the database
@st.cache_resource
def load_db():
    db_path = os.environ.get("QUOTES_DB_PATH")
    if not db_path:
        st.error("QUOTES_DB_PATH environment variable not set")
        st.stop()

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("quotes")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, model


collection, model = load_db()
# {{/docs-fragment load-db}}

# {{docs-fragment search-ui}}
# Search interface
query = st.text_input("Enter your search query:", placeholder="e.g., love, wisdom, success")
top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)

col1, col2 = st.columns([1, 1])
with col1:
    search_button = st.button("Search", type="primary", use_container_width=True)
with col2:
    random_button = st.button("Random Quote", use_container_width=True)

st.divider()
# {{/docs-fragment search-ui}}

# {{docs-fragment search-logic}}
if search_button and query:
    # Encode query and search
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    if results["documents"] and results["documents"][0]:
        for i, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            similarity = 1 - distance  # Convert distance to similarity
            st.markdown(f'**{i+1}.** "{doc}"')
            st.caption(f"— {metadata['author']} | Similarity: {similarity:.2%}")
            st.write("")
    else:
        st.info("No results found.")
# {{/docs-fragment search-logic}}

# {{docs-fragment random-quote}}
elif random_button:
    # Get a random quote from the collection
    all_data = collection.get(limit=100)
    if all_data["documents"]:
        idx = random.randint(0, len(all_data["documents"]) - 1)
        quote = all_data["documents"][idx]
        author = all_data["metadatas"][idx]["author"]
        st.markdown(f'**"{quote}"**')
        st.caption(f"— {author}")
# {{/docs-fragment random-quote}}

elif search_button and not query:
    st.warning("Please enter a search query.")
