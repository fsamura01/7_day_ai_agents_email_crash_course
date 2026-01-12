from data_preparation import read_repo_data


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
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break

    return result


def chunk_documents(docs, chunk_size=2000, step_size=1000):
    """
    Chunk multiple documents using sliding window approach.
    
    Args:
        docs: List of document dictionaries with 'content' field
        chunk_size: Size of each chunk in characters
        step_size: Step size between chunks
    
    Returns:
        List of chunked documents with metadata preserved
    """
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
    chunks = chunk_documents(docs, chunk_size=2000, step_size=1000)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Display sample chunk
    if chunks:
        print("\n" + "="*60)
        print("Sample Chunk:")
        print("="*60)
        sample = chunks[0]
        print(f"Filename: {sample.get('filename', 'N/A')}")
        print(f"Start position: {sample.get('start', 'N/A')}")
        print(f"Chunk length: {len(sample.get('chunk', ''))}")
        print(f"\nContent preview:\n{sample.get('chunk', '')[:200]}...")
        print("="*60)