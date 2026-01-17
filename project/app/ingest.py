import io
import zipfile
import requests
import frontmatter
import json
from pathlib import Path

def read_repo_data(repo_owner, repo_name):
    """
    Download and parse all markdown files from a GitHub repository.
    """
    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)
    
    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")
    
    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()
        
        if not (filename_lower.endswith('.md') or filename_lower.endswith('.mdx')):
            continue
        
        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(content)
                data = post.to_dict()
                
                # Strip top-level directory from zip filename for cleaner display and URL
                # Example: 'repo-main/README.md' -> 'README.md'
                clean_filename = '/'.join(filename.split('/')[1:])
                data['filename'] = clean_filename
                data['url'] = f"https://github.com/{repo_owner}/{repo_name}/blob/main/{clean_filename}"
                
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    zf.close()
    return repository_data

def chunk_documents(documents, chunk_size=2000, step_size=1000):
    """
    Split documents into overlapping chunks using a sliding window.
    """
    chunks = []
    for doc in documents:
        content = doc.get('content') or doc.get('text') or ""
        if not content and 'filename' in doc:
            # Fallback for docs where content is in a different key
            content = doc.get('body', '')
            
        if len(content) <= chunk_size:
            chunk = doc.copy()
            chunk['content'] = content.strip()
            chunk['start'] = 0
            chunks.append(chunk)
            continue
            
        for i in range(0, len(content), step_size):
            chunk_content = content[i : i + chunk_size]
            if not chunk_content:
                break
                
            chunk = doc.copy()
            chunk['content'] = chunk_content
            chunk['start'] = i
            chunks.append(chunk)
            
            if i + chunk_size >= len(content):
                break
                
    return chunks

def save_chunks(chunks, filepath="chunks.json"):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {filepath}")

def run_ingestion(repo_owner, repo_name, chunk_size=2000, step_size=1000):
    """
    Runs the full ingestion pipeline: Fetch -> Chunk -> Embed -> Save
    """
    print(f"Fetching from {repo_owner}/{repo_name}...")
    docs = read_repo_data(repo_owner, repo_name)
    
    print("Chunking...")
    chunks = chunk_documents(docs, chunk_size, step_size)
    save_chunks(chunks)
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        print("Loading embedding model...")
        model = SentenceTransformer('multi-qa-distilbert-cos-v1')
        
        print("Generating embeddings...")
        texts = [f"{c.get('content', '')}" for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        
        np.save("embeddings.npy", embeddings)
        print("Saved embeddings.npy")
    except Exception as e:
        import traceback
        print("\n" + "!"*50)
        print(f"CRITICAL: Embedding generation failed: {e}")
        print(traceback.format_exc())
        print("Continuing with KEYWORD SEARCH ONLY.")
        print("!"*50 + "\n")
