#!/usr/bin/env python3
"""
GitHub Repository Documentation Processor

This script fetches markdown documentation from a GitHub repository
and chunks it for processing by AI agents.

Usage:
    python run_pipeline.py <repo_owner> <repo_name> [--chunk-size SIZE] [--step-size SIZE]

Example:
    python run_pipeline.py fsamura01 task-manager-app
    python run_pipeline.py openai gpt-3 --chunk-size 3000 --step-size 1500
"""

import argparse
import json
from ingest import read_repo_data
from chunking import chunk_documents
from search import Index, VectorSearch, hybrid_search


def main():
    parser = argparse.ArgumentParser(
        description='Fetch and chunk GitHub repository documentation'
    )
    parser.add_argument('repo_owner', help='GitHub repository owner/organization')
    parser.add_argument('repo_name', help='GitHub repository name')
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=2000,
        help='Size of each chunk in characters (default: 2000)'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=1000,
        help='Step size between chunks (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start an interactive search session after chunking'
    )
    
    args = parser.parse_args()
    
    # Fetch repository data
    print(f"Fetching data from {args.repo_owner}/{args.repo_name}...")
    try:
        docs = read_repo_data(args.repo_owner, args.repo_name)
        print(f"Retrieved {len(docs)} documents")
    except Exception as e:
        print(f"Error fetching repository: {e}")
        return 1
    
    # Chunk documents
    print(f"\nChunking documents (size={args.chunk_size}, step={args.step_size})...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, step_size=args.step_size)
    print(f"Created {len(chunks)} chunks")
    
    # Display statistics
    if args.verbose and chunks:
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        
        total_chars = sum(len(c.get('content', '')) for c in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        print(f"Total chunks: {len(chunks)}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        # Sample chunk
        print("\n" + "="*70)
        print("SAMPLE CHUNK")
        print("="*70)
        sample = chunks[0]
        print(f"Filename: {sample.get('filename', 'N/A')}")
        print(f"Start position: {sample.get('start', 'N/A')}")
        print(f"Content length: {len(sample.get('content', ''))}")
        print(f"\nContent preview:\n{sample.get('content', '')[:300]}...")
        print("="*70)
    
    # Save to file if requested
    if args.output:
        print(f"\nSaving chunks to {args.output}...")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(chunks)} chunks to {args.output}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return 1
    
    print("\nPipeline completed successfully!")
    
    if args.interactive:
        print("\n" + "="*70)
        print("INTERACTIVE SEARCH MODE")
        print("="*70)
        print("Initializing search indexes...")
        
        # Initialize Index
        index = Index(text_fields=["content", "filename"])
        index.fit(chunks)
        
        # Initialize Vector Search (optional but recommended)
        v_index = None
        embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            import os
            
            print("Loading embedding model...")
            embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
            
            v_index = VectorSearch()
            # If we saved them before, load them. Otherwise, compute.
            embeddings_path = "embeddings.npy"
            if os.path.exists(embeddings_path):
                print(f"Loading pre-computed embeddings from {embeddings_path}")
                embeddings = np.load(embeddings_path)
                if len(embeddings) == len(chunks):
                    v_index.fit(embeddings, chunks)
                else:
                    print("Chunk count mismatch. Re-computing embeddings...")
                    v_index = None
            
            if v_index is None:
                v_index = VectorSearch()
                print("Computing embeddings (this may take a moment)...")
                texts = [f"{c.get('category', '')} {c.get('topic', '')} {c.get('content', '')}" for c in chunks]
                embeddings = embedding_model.encode(texts, show_progress_bar=True)
                v_index.fit(embeddings, chunks)
                np.save(embeddings_path, embeddings)
                print(f"Embeddings saved to {embeddings_path}")
                
        except Exception as e:
            print(f"Vector search initialization failed: {e}")
            print("Falling back to text-only search.")

        print("\nTypes of search available:")
        print("1. Keyword Search (Fast exact matches)")
        print("2. Video Search (Semantic/Conceptual matches)")
        print("Ready! Type your query below (or 'exit' to quit).")
        
        while True:
            query = input("\nSearch > ").strip()
            if not query or query.lower() in ['exit', 'quit', 'q']:
                break
                
            if v_index and embedding_model:
                results = hybrid_search(query, index, v_index, embedding_model, num_results=3)
                search_type = "Hybrid"
            else:
                results = index.search(query, num_results=3)
                search_type = "Text-only"
                
            print(f"\n--- {search_type} Search Results for '{query}' ---")
            if not results:
                print("No results found.")
                continue
                
            for i, res in enumerate(results, 1):
                print(f"\nResult {i}: {res.get('filename')}")
                content = res.get('content') or res.get('section') or res.get('chunk', '')
                print(f"Content: {content[:300]}...")
                print("-" * 20)
                
    return 0


if __name__ == '__main__':
    exit(main())
