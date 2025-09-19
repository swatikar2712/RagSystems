import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import RagPipeline

# Initialize the RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    return RagPipeline()

rag_pipeline = load_rag_pipeline()

# App layout
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📚 Retrieval-Augmented Generation (RAG) App")
st.markdown("Ask a question based on the **AI Training Document**.")

# Sidebar model info
st.sidebar.header("Model Info")
st.sidebar.write(f"**LLM Model:** {rag_pipeline.model_name}")

# Session state initialization
if "query" not in st.session_state:
    st.session_state.query = ""

# Query input box
st.session_state.query = st.text_input(
    "Enter your query:",
    value=st.session_state.query,
    placeholder="e.g., What is eBay's policy on returns?"
)

# Clear chat button below the query box
if st.button("🧹 Clear Chat"):
    st.session_state.query = ""
    st.rerun()

# Only run RAG if query exists
if st.session_state.query:
    with st.spinner("Generating answer..."):
        context, streamer = rag_pipeline.Pipeline(st.session_state.query)

        # Dynamic output container
        output_box = st.empty()
        full_answer = ""

        # Stream tokens and update output
        for token in streamer:
            full_answer += token
            if "[/INST]" in full_answer:
                full_answer = full_answer.split("[/INST]")[-1].strip()
            output_box.markdown(f"**{full_answer}**")

    # Show the retrieved context
    with st.expander("📄 Context used from the document"):
        st.write(context)
