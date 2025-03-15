import os
import numpy as np
import faiss
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datetime import datetime

# 1. Data Collection & Preprocessing

def read_reports(directory="financial_reports/"):
    all_text = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                all_text.append(file.read())
    return all_text

documents = read_reports()

# 2. Basic RAG Implementation
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=100):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc))

embeddings = embed_model.encode(chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# 3. Advanced RAG Implementation (Chunk Merging & Adaptive Retrieval)
def merge_chunks(chunks, window_size=2):
    merged_chunks = []
    for i in range(len(chunks) - window_size + 1):
        merged_chunks.append(" ".join(chunks[i:i+window_size]))
    return merged_chunks

merged_chunks = merge_chunks(chunks)
merged_embeddings = embed_model.encode(merged_chunks, convert_to_numpy=True)
merged_index = faiss.IndexFlatL2(merged_embeddings.shape[1])
merged_index.add(merged_embeddings)
merged_chunk_map = {i: chunk for i, chunk in enumerate(merged_chunks)}

def chunk_text_by_year(text):
    # Split text based on patterns like "Year: XXXX" or "XXXX Revenue:"
    year_pattern = re.compile(r'(Year:\s*\d{4}|20\d{2}\s*Revenue)')
    split_text = re.split(year_pattern, text)

    # Combine the year header with its content for clear segmentation
    chunks = []
    for i in range(1, len(split_text), 2):
        year_chunk = split_text[i] + split_text[i + 1] if i + 1 < len(split_text) else split_text[i]
        chunks.append(year_chunk.strip())
    return chunks

chunks = []
for doc in documents:
    chunks.extend(chunk_text_by_year(doc))

# Mapping and Embedding
embeddings = embed_model.encode(chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

def get_year_from_query(query):
    current_year = datetime.now().year
    query_lower = query.lower()

    # Regex to match a specific year (e.g., "2023", "2024")
    year_match = re.search(r'\b(20\d{2})\b', query_lower)
    if year_match:
        return int(year_match.group(0))

    # Handling relative year queries
    if "last year" in query_lower:
        return current_year - 1
    elif "this year" in query_lower:
        return current_year
    elif "next year" in query_lower:
        return current_year + 1
    elif "years back" in query_lower or "years ago" in query_lower:
        for word in query_lower.split():
            if word.isdigit():
                return current_year - int(word)

    # If no year or relative term is detected
    return None

# Enhanced Year Filtering
def adaptive_retrieval(query, top_k=3, advanced=False):
    query_year = get_year_from_query(query)
    
    finance_keywords = ["revenue", "profit", "earnings", "financial"]
    if not any(word in query.lower() for word in finance_keywords):
        return "This query is not related to financial data."
    
    if query_year is None:
        return "Please specify relevant year or use terms like 'last year' or 'X years back'."

    query_embedding = embed_model.encode([query])
    _, faiss_results = index.search(query_embedding, top_k)
    valid_indices = [idx for idx in faiss_results[0] if idx != -1 and idx in chunk_map]
    
    retrieved_chunks = [chunk_map[idx] for idx in valid_indices]

    # Strictly match target year data
    filtered_chunks = [
        chunk for chunk in retrieved_chunks
        if re.search(rf'\b{query_year}\b', chunk) and not re.search(rf'\b{query_year + 1}\b', chunk)
    ]

    # Clean response text
    cleaned_chunks = [chunk.replace("\n", "").replace(" .", ".").strip() for chunk in filtered_chunks]

    if not cleaned_chunks:
        return f"No data found for the year {query_year}."

    response = f"Here is the information I found for the year {query_year}:\n"
    for chunk in cleaned_chunks:
        response += f"- {chunk}\n"

    return response

# 5. Guard Rail Implementation
def filter_response(response):
    banned_phrases = ["I think", "maybe", "possibly", "I'm not sure"]
    for phrase in banned_phrases:
        if phrase in response:
            return "Filtered due to low confidence."
    return response

# 4. UI Development with Streamlit
st.title("Financial Report RAG System")
query = st.text_input("Enter your financial query:")
if query:
    st.subheader("Basic RAG Response:")
    basic_response = adaptive_retrieval(query, advanced=False)
    st.write(basic_response)
    
    st.subheader("Advanced RAG Response (Chunk Merging & Adaptive Retrieval):")
    advanced_response = adaptive_retrieval(query, advanced=True)
    st.write(advanced_response)