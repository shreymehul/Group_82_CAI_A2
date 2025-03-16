import os
import numpy as np
import faiss
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources (run once)
nltk.download("punkt")
nltk.download("stopwords")

# Constants
CHUNK_SIZES = {"small": 50, "medium": 100, "large": 200}
FINANCE_KEYWORDS = ["revenue", "profit", "earnings", "financial"]

# Initialize SentenceTransformer model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def read_reports(directory="financial_reports/"):
    """Read all financial reports from the specified directory."""
    return [open(os.path.join(directory, filename), "r").read() 
            for filename in os.listdir(directory) if filename.endswith(".txt")]

def chunk_text(text, chunk_size=100):
    """Split text into chunks of a specified size."""
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def preprocess_text(text):
    """Preprocess text for BM25: lowercase, remove stopwords, and stem (except numbers)."""
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) if not token.isdigit() else token
              for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def merge_chunks(chunks, query_year, window_size=2):
    """Merge chunks that belong to the same year."""
    merged_chunks = []
    i = 0
    while i < len(chunks):
        if re.search(rf"\b{query_year}\b", chunks[i]):
            merged_chunk = chunks[i]
            for j in range(1, window_size):
                if i + j < len(chunks) and re.search(rf"\b{query_year}\b", chunks[i + j]):
                    merged_chunk += " " + chunks[i + j]
                else:
                    break
            merged_chunks.append(merged_chunk)
            i += window_size
        else:
            i += 1
    return merged_chunks

def rerank_chunks(query, retrieved_chunks, top_k=3):
    """Re-rank retrieved chunks based on cosine similarity."""
    query_embedding = embed_model.encode([query])
    chunk_embeddings = embed_model.encode(retrieved_chunks)
    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return [retrieved_chunks[i] for i in ranked_indices]

def compute_bm25_confidence(bm25_scores):
    """Normalize BM25 scores to a confidence range of [0,1]."""
    if bm25_scores.size == 0:
        return np.array([])
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()

def compute_embedding_confidence(query, retrieved_chunks):
    """Compute cosine similarity between query embedding and retrieved chunks."""
    query_embedding = embed_model.encode([query])
    chunk_embeddings = embed_model.encode(retrieved_chunks)
    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    if len(similarities) == 1:
        return np.array([1.0])
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(similarities.reshape(-1, 1)).flatten()

def compute_final_confidence(bm25_confidences, embedding_confidences, weight_bm25=0.5, weight_embed=0.5):
    """Compute final confidence score by weighted fusion of BM25 and embedding confidences."""
    if bm25_confidences.size == 0 and embedding_confidences.size == 0:
        return np.array([])
    elif bm25_confidences.size == 0 or np.all(bm25_confidences == 0):
        return embedding_confidences
    elif embedding_confidences.size == 0:
        return bm25_confidences
    return (weight_bm25 * bm25_confidences) + (weight_embed * embedding_confidences)

def get_year_from_query(query):
    """Extract the year from the query."""
    current_year = datetime.now().year
    query_lower = query.lower()
    year_match = re.search(r"\b(20\d{2})\b", query_lower)
    if year_match:
        return int(year_match.group(0))
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
    return None

def adaptive_retrieval(query, top_k=3, advanced=False):
    """Retrieve and rank chunks based on the query."""
    query_year = get_year_from_query(query)
    if not any(word in query.lower() for word in FINANCE_KEYWORDS):
        return "This query is not related to financial data."
    if not query_year:
        return "Please specify relevant year or use terms like 'last year' or 'X years back'."

    if advanced:
        tokenized_query = preprocess_text(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        retrieved_chunks = [chunks[i] for i in top_bm25_indices]
        merged_chunks = merge_chunks(retrieved_chunks, query_year)
        bm25_confidences = compute_bm25_confidence(np.array([bm25_scores[i] for i in top_bm25_indices]))
        embedding_confidences = compute_embedding_confidence(query, merged_chunks)
        final_confidences = compute_final_confidence(bm25_confidences, embedding_confidences)
        reranked_chunks = rerank_chunks(query, merged_chunks, top_k)
    else:
        query_embedding = embed_model.encode([query])
        _, faiss_results = index.search(query_embedding, top_k)
        valid_indices = [idx for idx in faiss_results[0] if idx != -1 and idx in chunk_map]
        reranked_chunks = [chunk_map[idx] for idx in valid_indices]
        final_confidences = compute_embedding_confidence(query, reranked_chunks)

    filtered_chunks = [chunk for chunk in reranked_chunks if re.search(rf"\b{query_year}\b", chunk)]
    cleaned_chunks = [chunk.replace("\n", "").replace(" .", ".").strip() for chunk in filtered_chunks]

    if not cleaned_chunks:
        return f"No data found for the year {query_year}."

    response = f"Here is the information I found for the year {query_year}:\n"
    for i, chunk in enumerate(cleaned_chunks):
        confidence = final_confidences[i] if i < len(final_confidences) else 0.5
        response += f"- **{chunk}**\n  (Confidence: {confidence:.2f})\n"
    return response

# Load documents and preprocess
documents = read_reports()
chunks = [chunk for doc in documents for chunk in chunk_text(doc, CHUNK_SIZES["medium"])]
embeddings = embed_model.encode(chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
tokenized_chunks = [preprocess_text(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Streamlit UI
st.title("Financial Report RAG System")
query = st.text_input("Enter your financial query:")
if query:
    st.subheader("Basic RAG Response:")
    st.write(adaptive_retrieval(query, advanced=False))
    st.subheader("Advanced RAG Response (BM25 + Chunk Merging + Re-ranking):")
    st.write(adaptive_retrieval(query, advanced=True))