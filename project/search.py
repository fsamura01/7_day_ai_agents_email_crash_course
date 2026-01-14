import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class Index:
    def __init__(self, text_fields: List[str], keyword_fields: List[str] = None):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields or []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.docs = []
        self.tfidf_matrix = None

    def fit(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        # Combine text fields for each document
        corpus = []
        for doc in docs:
            text = " ".join([str(doc.get(f, "")) for f in self.text_fields])
            corpus.append(text)
        
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return self

    def search(self, query: str, num_results: int = 5, boost: Dict[str, float] = None) -> List[Dict[str, Any]]:
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
    # Get text search results
    text_results = index.search(query, num_results=num_results * 2)
    
    # Get vector search results
    query_embedding = embedding_model.encode(query)
    vector_results = vector_index.search(query_embedding, num_results=num_results * 2)
    
    # Combine and deduplicate
    seen = set()
    combined = []
    
    for res in text_results + vector_results:
        # Standardize on 'content' key
        content = res.get('content') or res.get('section') or res.get('chunk', '')
        key = (res.get('filename', ''), content)
        if key not in seen:
            seen.add(key)
            combined.append(res)
            
    return combined[:num_results]

if __name__ == "__main__":
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--data", type=str, default="chunks.json", help="Path to chunked data JSON")
    parser.add_argument("--embeddings", type=str, default="embeddings.npy", help="Path to embeddings numpy file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found. Run the pipeline first.")
    else:
        with open(args.data, 'r') as f:
            chunks = json.load(f)
            
        print(f"✓ Loaded {len(chunks)} chunks")
        
        # 1. Text Search
        print("\n--- Testing Text Search ---")
        index = Index(text_fields=["content", "filename"])
        index.fit(chunks)
        results = index.search(args.query, num_results=2)
        for i, res in enumerate(results, 1):
            print(f"Text Result {i} (Score: {res['score']:.4f}): {res.get('filename')}")

        # 2. Vector Search (if sentence-transformers is available)
        try:
            from sentence_transformers import SentenceTransformer
            print("\n--- Testing Vector Search ---")
            model = SentenceTransformer('multi-qa-distilbert-cos-v1')
            
            # Create or load embeddings
            if os.path.exists(args.embeddings):
                embeddings = np.load(args.embeddings)
                print(f"✓ Loaded embeddings from {args.embeddings}")
            else:
                print("Creating embeddings (this may take a moment)...")
                texts = [f"{c.get('category', '')} {c.get('topic', '')} {c.get('content', '')}" for c in chunks]
                embeddings = model.encode(texts, show_progress_bar=True)
                np.save(args.embeddings, embeddings)
                print(f"✓ Saved embeddings to {args.embeddings}")
            
            v_index = VectorSearch()
            v_index.fit(embeddings, chunks)
            
            query_emb = model.encode(args.query)
            v_results = v_index.search(query_emb, num_results=2)
            for i, res in enumerate(v_results, 1):
                print(f"Vector Result {i} (Score: {res['score']:.4f}): {res.get('filename')}")
                
            # 3. Hybrid Search
            print("\n--- Testing Hybrid Search ---")
            h_results = hybrid_search(args.query, index, v_index, model, num_results=3)
            for i, res in enumerate(h_results, 1):
                print(f"Hybrid Result {i}: {res.get('filename')} - {res.get('content', '')[:100]}...")
                
        except ImportError:
            print("\nNote: sentence-transformers not found. Skipping vector/hybrid search tests.")
        except Exception as e:
            print(f"\nError during vector search test: {e}")
