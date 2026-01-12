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
from data_preparation import read_repo_data
from chunking import chunk_documents


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
    
    args = parser.parse_args()
    
    # Fetch repository data
    print(f"📥 Fetching data from {args.repo_owner}/{args.repo_name}...")
    try:
        docs = read_repo_data(args.repo_owner, args.repo_name)
        print(f"✓ Retrieved {len(docs)} documents")
    except Exception as e:
        print(f"✗ Error fetching repository: {e}")
        return 1
    
    # Chunk documents
    print(f"\n🔪 Chunking documents (size={args.chunk_size}, step={args.step_size})...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, step_size=args.step_size)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Display statistics
    if args.verbose and chunks:
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        
        total_chars = sum(len(c.get('chunk', '')) for c in chunks)
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
        print(f"Chunk length: {len(sample.get('chunk', ''))}")
        print(f"\nContent preview:\n{sample.get('chunk', '')[:300]}...")
        print("="*70)
    
    # Save to file if requested
    if args.output:
        print(f"\n💾 Saving chunks to {args.output}...")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(chunks)} chunks to {args.output}")
        except Exception as e:
            print(f"✗ Error saving file: {e}")
            return 1
    
    print("\n✅ Pipeline completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
