import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import re

from dotenv import load_dotenv
import os

# Load Gemini API key from .env
load_dotenv()
gen_api_key = os.getenv("GOOGLE_API_KEY")

# Load models and data
embedding_model = SentenceTransformer("fine_tuned_quote_model")
index = faiss.read_index("quotes_index.faiss")
df = pd.read_csv("your_cleaned_quotes.csv")
genai.configure(api_key=gen_api_key)
llm = genai.GenerativeModel("gemini-1.5-flash-latest")

# Function: Search Quotes
def search_quotes(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    D, I = index.search(query_embedding, top_k)

    results = []
    for i in I[0]:
        quote_data = {
            "quote": df.iloc[i]['quote'],
            "author": df.iloc[i]['author'],
            "tags": df.iloc[i]['tags']
        }
        results.append(quote_data)
    return results

# Function: Generate LLM Response
def generate_structured_response(query, retrieved_quotes):
    context = "\n".join([
        f"{r['quote']} — {r['author']} (Tags: {', '.join(r['tags'])})"
        for r in retrieved_quotes
    ])

    prompt = f"""
You are a smart assistant. Use the quotes below to answer the query.

Query: "{query}"

Quotes:
{context}

Return a structured JSON with:
- quotes
- authors
- tags
- summary
"""
    response = llm.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Quote Search with RAG")

query = st.text_input("Enter your quote query")

if st.button("Search"):
    if query.strip():
        with st.spinner("Retrieving quotes..."):
            results = search_quotes(query)
            response = generate_structured_response(query, results)

            # Show retrieved quotes
            st.subheader("🔎 Retrieved Quotes:")
            for i, r in enumerate(results):
                st.markdown(f"**{i+1}.** _{r['quote']}_ — **{r['author']}**")

            # Show raw response
            st.subheader("Gemini Response (Raw Text):")
            st.text(response)
            
            clean_text = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()

            # Try to parse as JSON and display nicely
            try:
                parsed = json.loads(clean_text)
                st.subheader("Parsed JSON:")
                st.json(parsed)
            except json.JSONDecodeError:
                st.warning("Response is not valid JSON. Displaying raw text.")

            # Always offer download of raw response
            raw_json = json.dumps({"response": response}, indent=4)
            st.download_button(
                label="Download Response as JSON",
                data=raw_json,
                file_name="gemini_response.json",
                mime="application/json"
            )

    else:
        st.warning("Please enter a query.")
