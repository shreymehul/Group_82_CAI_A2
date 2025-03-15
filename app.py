import os
import numpy as np
import faiss
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

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

# Generate chunks of different sizes for testing
chunks_small = []
chunks_medium = []
chunks_large = []
for doc in documents:
    chunks_small.extend(chunk_text(doc, chunk_size=50))  # Small chunks
    chunks_medium.extend(chunk_text(doc, chunk_size=100))  # Medium chunks
    chunks_large.extend(chunk_text(doc, chunk_size=200))  # Large chunks

# Use medium chunks for Basic RAG
chunks = chunks_medium
embeddings = embed_model.encode(chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

# 3. Advanced RAG Implementation
# BM25 for keyword-based search
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Chunk Merging Function
# Chunk Merging Function (Year-Aware)
def merge_chunks(chunks, query_year, window_size=2):
    merged_chunks = []
    i = 0
    while i < len(chunks):
        # Check if the current chunk belongs to the query year
        if re.search(rf'\b{query_year}\b', chunks[i]):
            merged_chunk = chunks[i]
            # Merge subsequent chunks if they belong to the same year
            for j in range(1, window_size):
                if i + j < len(chunks) and re.search(rf'\b{query_year}\b', chunks[i + j]):
                    merged_chunk += " " + chunks[i + j]
                else:
                    break
            merged_chunks.append(merged_chunk)
            i += window_size  # Skip the merged chunks
        else:
            i += 1  # Move to the next chunk
    return merged_chunks

# Re-ranking function
def rerank_chunks(query, retrieved_chunks, top_k=3):
    query_embedding = embed_model.encode([query])
    chunk_embeddings = embed_model.encode(retrieved_chunks)
    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return [retrieved_chunks[i] for i in ranked_indices]

# Adaptive retrieval with BM25, Chunk Merging, and Re-ranking
def adaptive_retrieval(query, top_k=3, advanced=False):
    query_year = get_year_from_query(query)
    
    finance_keywords = ["revenue", "profit", "earnings", "financial"]
    if not any(word in query.lower() for word in finance_keywords):
        return "This query is not related to financial data."
    
    if query_year is None:
        return "Please specify relevant year or use terms like 'last year' or 'X years back'."

    if advanced:
        # Advanced RAG: BM25 + Year-Aware Chunk Merging + Re-ranking
        # Step 1: BM25 for keyword-based retrieval
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]  # Retrieve more chunks for merging and re-ranking
        retrieved_chunks = [chunks[i] for i in top_bm25_indices]

        # Step 2: Merge chunks for better context (year-aware merging)
        merged_chunks = merge_chunks(retrieved_chunks, query_year, window_size=2)

        # Step 3: Re-rank using embeddings
        reranked_chunks = rerank_chunks(query, merged_chunks, top_k)
    else:
        # Basic RAG: Embedding-based retrieval
        query_embedding = embed_model.encode([query])
        _, faiss_results = index.search(query_embedding, top_k)
        valid_indices = [idx for idx in faiss_results[0] if idx != -1 and idx in chunk_map]
        reranked_chunks = [chunk_map[idx] for idx in valid_indices]

    # Filter by year
    filtered_chunks = [
        chunk for chunk in reranked_chunks
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

# Helper function to extract year from query
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

# 4. Guard Rail Implementation
def filter_response(response):
    banned_phrases = ["I think", "maybe", "possibly", "I'm not sure"]
    for phrase in banned_phrases:
        if phrase in response:
            return "Filtered due to low confidence."
    return response

# 5. UI Development with Streamlit
st.title("Financial Report RAG System")
query = st.text_input("Enter your financial query:")
if query:
    st.subheader("Basic RAG Response:")
    basic_response = adaptive_retrieval(query, advanced=False)
    st.write(basic_response)
    
    st.subheader("Advanced RAG Response (BM25 + Chunk Merging + Re-ranking):")
    advanced_response = adaptive_retrieval(query, advanced=True)
    st.write(advanced_response)