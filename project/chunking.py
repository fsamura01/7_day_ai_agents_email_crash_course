from ingest import read_repo_data
from intelligent_chunking import process_documents_intelligent, setup_llm_client


def sliding_window(seq, size, step):
    """
    Create overlapping chunks from a sequence using a sliding window approach.
    
    Args:
        seq: The sequence to chunk (e.g., string, list)
        size: Size of each chunk
        step: Step size between chunks (overlap = size - step)
    
    Returns:
        List of dictionaries with 'start' position and 'chunk' content
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'content': chunk})
        if i + size >= n:
            break

    return result


def chunk_documents(docs, method='intelligent', provider='groq', chunk_size=2000, step_size=1000):
    """
    Chunk multiple documents using either intelligent or sliding window approach.
    
    Args:
        docs: List of document dictionaries with 'content' field
        method: 'intelligent' or 'sliding_window'
        provider: 'groq' or 'openai' (for intelligent chunking)
        chunk_size: Size of each chunk in characters (for sliding window)
        step_size: Step size between chunks (for sliding window)
    
    Returns:
        List of chunked documents with metadata preserved
    """
    if method == 'intelligent':
        try:
            return process_documents_intelligent(docs, provider=provider)
        except Exception as e:
            print(f"Intelligent chunking failed, falling back to sliding window: {e}")
            # Continue to sliding window fallback below
    
    all_chunks = []
    
    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content', '')
        
        if not doc_content:
            print(f"Warning: No content found in {doc_copy.get('filename', 'unknown file')}")
            continue
        
        chunks = sliding_window(doc_content, chunk_size, step_size)
        
        for chunk in chunks:
            # Add document metadata to each chunk
            chunk.update(doc_copy)
            chunk['chunking_method'] = 'sliding_window'
        
        all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == '__main__':
    # Example usage
    repo_owner = 'fsamura01'
    repo_name = 'task-manager-app'
    
    print(f"Fetching data from {repo_owner}/{repo_name}...")
    docs = read_repo_data(repo_owner, repo_name)
    print(f"✓ Retrieved {len(docs)} documents")
    
    print("\nChunking documents...")
    # By default uses intelligent chunking if GROQ_API_KEY is set
    chunks = chunk_documents(docs, method='intelligent')
    print(f"✓ Created {len(chunks)} chunks")
    
    # Display sample chunk
    if chunks:
        print("\n" + "="*60)
        print("Sample Chunk:")
        print("="*60)
        sample = chunks[0]
        print(f"Filename: {sample.get('filename', 'N/A')}")
        print(f"Chunking Method: {sample.get('chunking_method', 'N/A')}")
        
        # Use standardized 'content' key
        content = sample.get('content') or sample.get('section') or ''
        
        print(f"Content length: {len(content)}")
        print(f"\nContent preview:\n{content[:200]}...")
        print("="*60)