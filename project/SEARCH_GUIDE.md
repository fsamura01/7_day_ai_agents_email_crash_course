# Hybrid Search Guide

This guide explains the logic behind the search functionality in the project, including how it works and how to interpret the results.

## Overview: What is Hybrid Search?
Hybrid search combines two different ways of looking for information to give you the most relevant results:
1. Keyword Search (Text): Finds exact word matches.
2. Semantic Search (Vector): Finds similar meanings and concepts.

---

## 1. Keyword Search (Index Class)
This is traditional search, like a book index or Ctrl+F. It uses TF-IDF and Cosine Similarity.

Logic: It calculates how unique a word is. Common words like "the" are ignored, while technical words like "PostgreSQL" or "Endpoint" get high priority.

Best for: Specific technical terms (Node.js, API), identifying exact phrases from headers, and quick literal lookups.

Limitation: If you have a typo or use a synonym (e.g., "how to start" vs "installation"), it might not find the result.

## 2. Semantic Search (VectorSearch Class)
This uses AI Embeddings to understand the intent behind your query.

Logic: Your text is converted into a list of numbers (a vector) that represents its meaning. The engine then looks for other vectors that are close in mathematical space.

Best for: Natural language questions, conceptual searches, and handling synonyms.

The Cache (embeddings.npy): These vectors are saved to a file to make searching instant.

---

## 3. The Hybrid Search Brain
The hybrid_search function combines both methods. First, it runs a keyword search to get exact matches. Second, it runs a semantic search to get meaning matches. Third, it merges both lists and removes duplicates. Finally, it returns the top-ranked results.

---

## 4. Step-by-Step Query Processing
Let's trace a query like "What are the API requirements?" through the system.

Phase A: Initialization and Fitting
The system scans your documents and calculates the importance of every word to build a mathematical Keyword Map.

Phase B: Keyword Processing
Your query is stripped of common words. The remaining words (API, requirements) are compared to all documents to find literal matches.

Phase C: Semantic Processing
An AI model turns your query into a meaning vector. It then looks for chunks in the embeddings.npy file that are conceptually similar.

Phase D: The Hybrid Merge
Results from both Keyword and Semantic phases are combined. Duplicates are removed, and the best results are shown to you.

---

## 5. Understanding the Output
When you run a search, you will see the rank of the result, the filename where it was found, and the content snippet. The Intelligent Chunking ensures these snippets start with relevant headers.

---

## 6. How to Run

Interactive Mode
Run the pipeline with the --interactive flag:
python project/run_pipeline.py repo_owner repo_name --interactive

Standalone CLI
Use the search utility directly:
python project/search.py --query "your question here"
