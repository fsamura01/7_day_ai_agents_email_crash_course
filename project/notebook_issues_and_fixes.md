# Notebook Debugging & Optimization Guide: `notebook-Copy1.ipynb`

This document provides a comprehensive analysis of the issues identified in your notebook (starting from cell 100) and provides ready-to-use fixes and advanced search optimizations.

---

## 🔴 Part 1: Quick Fixes (Apply These First)

### 1. Variable Name Error (Cell 103)
**Issue:** `NameError: name 'task_manager_chunks' is not defined`
**Fix:** Replace all instances of `task_manager_chunks` with `task_manager_app_chunks` in cell 103 (lines 816 and 819).

### 2. Missing `VectorSearch` Class
**Issue:** `NameError: name 'VectorSearch' is not defined`
**Fix:** Insert a new cell before your vector search implementation with this code:

```python
import numpy as np

class VectorSearch:
    def __init__(self):
        self.vectors = None
        self.documents = None
    
    def fit(self, vectors, documents):
        self.vectors = vectors
        self.documents = documents
    
    def search(self, query_vector, num_results=5):
        # Normalize vectors for cosine similarity
        query_norm = query_vector / np.linalg.norm(query_vector)
        v_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1 # Avoid division by zero
        vectors_norm = self.vectors / v_norms
        
        # Calculate scores
        similarities = np.dot(vectors_norm, query_norm)
        top_indices = np.argsort(similarities)[::-1][:num_results]
        return [self.documents[i] for i in top_indices]
```

### 3. Fixing Empty Category Results (Cell 113)
**Issue:** Search by category returns **0 results** because metadata is missing.
**Fix:** Update your chunking loop to include auto-categorization:

```python
def auto_categorize(chunk_text, filename):
    text = chunk_text.lower()
    fname = filename.lower()
    
    # Category rules
    category = 'Frontend' if 'client' in fname or 'react' in text or 'component' in text else \
               'API' if 'api' in text or 'endpoint' in text else \
               'Backend' if 'database' in text or 'schema' in text or 'postgresql' in text else 'General'
    
    # Topic rules
    topic = 'Validation' if 'validation' in text else \
            'Errors' if 'error' in text else \
            'Setup' if 'install' in text or 'setup' in text else 'Overview'
    
    return category, topic

# apply during chunking...
for chunk in chunks:
    chunk.update(doc_copy)
    cat, top = auto_categorize(chunk['chunk'], chunk['filename'])
    chunk['category'], chunk['topic'] = cat, top
```

---

## 🟠 Part 2: Advanced Search Optimizations

### 1. Hybrid Search with Proper Ranking
Instead of just combining lists, use a weighted scoring system to merge results from Keyword (Text) and Semantic (Vector) searches.

```python
def search_hybrid_ranked(query, num_results=5, text_weight=0.5, vector_weight=0.5):
    # Get 2x results for merging
    n = num_results * 2
    text_results = text_search.search(query, num_results=n)
    q_emb = embedding_model.encode(query)
    vector_results = vector_search.search(q_emb, num_results=n)
    
    scores = {}
    # Higher rank = higher weight
    for i, res in enumerate(text_results):
        key = (res['start'], res['filename'])
        scores[key] = scores.get(key, 0) + text_weight * (n - i) / n
    
    for i, res in enumerate(vector_results):
        key = (res['start'], res['filename'])
        scores[key] = scores.get(key, 0) + vector_weight * (n - i) / n
    
    # Sort by combined score
    sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    res_map = {(r['start'], r['filename']): r for r in text_results + vector_results}
    
    return [res_map[key] for key, _ in sorted_keys[:num_results]]
```

### 2. Search Analytics
Add a utility to see how your different search methods overlap:

```python
def analyze_search(query):
    text_keys = {(r['start'], r['filename']) for r in text_search.search(query, 5)}
    q_emb = embedding_model.encode(query)
    vector_keys = {(r['start'], r['filename']) for r in vector_search.search(q_emb, 5)}
    
    overlap = len(text_keys & vector_keys)
    print(f"Query: '{query}'")
    print(f"Overlap between Keyword & Semantic: {overlap}/5")
```

---

## 🟢 Part 3: Testing Checklist
- [ ] Run the `VectorSearch` class cell.
- [ ] Re-run chunking with `auto_categorize`.
- [ ] Verify `text_search.fit(task_manager_app_chunks)` executes.
- [ ] Verify `search_by_category("validation", "Backend")` now returns results.
- [ ] Run `search_hybrid_ranked` for queries like "How do I add a task?".
