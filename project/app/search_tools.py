import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import os
import json

class Index:
    def __init__(self, text_fields: List[str]):
        self.text_fields = text_fields
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.docs = []
        self.tfidf_matrix = None

    def fit(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        corpus = []
        for doc in docs:
            text = " ".join([str(doc.get(f, "")) for f in self.text_fields])
            corpus.append(text)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return self

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-num_results:][::-1]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                doc = self.docs[i].copy()
                doc['score'] = float(scores[i])
                results.append(doc)
        return results

class VectorSearch:
    def __init__(self):
        self.embeddings = None
        self.docs = []

    def fit(self, embeddings: np.ndarray, docs: List[Dict[str, Any]]):
        self.embeddings = embeddings
        self.docs = docs
        return self

    def search(self, query_embedding: np.ndarray, num_results: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            return []
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = np.argsort(scores)[-num_results:][::-1]
        results = []
        for i in top_indices:
            doc = self.docs[i].copy()
            doc['score'] = float(scores[i])
            results.append(doc)
        return results

def hybrid_search(query: str, index: Index, vector_index: VectorSearch, embedding_model, num_results: int = 5) -> List[Dict[str, Any]]:
    # Keyword search
    text_results = index.search(query, num_results=num_results * 2)
    # Vector search
    query_embedding = embedding_model.encode(query)
    vector_results = vector_index.search(query_embedding, num_results=num_results * 2)
    
    # Simple combine (deduplicate)
    seen = set()
    combined = []
    for res in text_results + vector_results:
        # Use filename and content snippet as key for deduplication
        key = (res.get('filename', ''), res.get('content', '')[:100])
        if key not in seen:
            seen.add(key)
            combined.append(res)
    return combined[:num_results]

# --- Global State for the Tool ---
_index = None
_v_index = None
_model = None

def initialize_search_indexes(chunks_path="chunks.json", embeddings_path="embeddings.npy"):
    global _index, _v_index, _model
    
    # Avoid reloading if already initialized
    if _index is not None and _model is not None:
        return True
        
    if not os.path.exists(chunks_path):
        return False
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    _index = Index(text_fields=["content", "filename"]).fit(chunks)
    
    if os.path.exists(embeddings_path):
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('multi-qa-distilbert-cos-v1')
        embeddings = np.load(embeddings_path)
        _v_index = VectorSearch().fit(embeddings, chunks)
    
    return True

# --- The Pydantic AI Agent Tool ---
def text_search(query: str) -> str:
    """
    Search the repository documentation using keywords and semantic similarity.
    """
    if _index is None:
        return "Search index not initialized. Please run ingestion first."
        
    try:
        if _v_index and _model:
            results = hybrid_search(query, _index, _v_index, _model, num_results=2)
        else:
            results = _index.search(query, num_results=2)
            
        if not results:
            return "No relevant documentation found."
            
        formatted = []
        for i, res in enumerate(results, 1):
            source = res.get('filename', 'Unknown')
            content = res.get('content', 'No content')
            snippet = content[:600] + "..." if len(content) > 600 else content
            formatted.append(f"Source {i} [{source}]:\n{snippet}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error during search: {str(e)}"
