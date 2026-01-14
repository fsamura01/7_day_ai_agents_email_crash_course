import os
import json
import asyncio
import numpy as np
from typing import List, Any, Optional
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer
from search import Index, VectorSearch, hybrid_search

# --- Configuration & Shared State ---
# These paths point to the data created by run_pipeline.py
CHUNK_FILE = "chunks.json"
EMBEDDING_FILE = "embeddings.npy"

# Global search components
index: Optional[Index] = None
v_index: Optional[VectorSearch] = None
model: Optional[SentenceTransformer] = None

def initialize_search():
    """
    Initializes search indexes by loading cached data from the project directory.
    If chunks.json or embeddings.npy are missing, the search will fall back to
    limited mode or tool-level warnings.
    """
    global index, v_index, model
    
    # Check for the primary cache file (chunks.json)
    if os.path.exists(CHUNK_FILE):
        try:
            with open(CHUNK_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # 1. Initialize Keyword Index (TF-IDF)
            index = Index(text_fields=["content", "filename"])
            index.fit(chunks)
            print(f"[OK] Loaded keyword index from {CHUNK_FILE}")
            
            # 2. Initialize Vector Index if embeddings exist
            if os.path.exists(EMBEDDING_FILE):
                print("Loading embedding model and vectors...")
                model = SentenceTransformer('multi-qa-distilbert-cos-v1')
                embeddings = np.load(EMBEDDING_FILE)
                
                v_index = VectorSearch()
                v_index.fit(embeddings, chunks)
                print(f"[OK] Loaded semantic index from {EMBEDDING_FILE}")
            else:
                print("[WARNING] Semantic cache (embeddings.npy) missing. Agent will use Text-only search.")
                
        except Exception as e:
            print(f"Error initializing indexes: {e}")
    else:
        print(f"Warning: {CHUNK_FILE} not found. The agent will not have access to repo data.")

# Run initialization during module load
initialize_search()

# --- Agent Definition ---

SYSTEM_PROMPT = """
You are a helpful technical support agent for the GitHub repository documentation.
Your goal is to answer user questions regarding the project, its API, and its architecture.

RULES:
1. ALWAYS use the 'text_search' tool to look up information before answering.
2. Only answer based on the search results provided by the tool.
3. If the tool returns no data or says the index is missing, tell the user they need to run the data collection pipeline.
4. If you cannot find the answer in the search results, say "I don't have enough information in the current documentation to answer that."
"""

# Initialize the Pydantic AI Agent
# Note: Ensure GROQ_API_KEY is set in your environment
agent = Agent(
    'groq:llama-3.1-8b-instant',
    name="repo_documentation_agent",
    system_prompt=SYSTEM_PROMPT,
)

@agent.tool_plain
def text_search(query: str) -> str:
    """
    Analyzes the repository's documentation using a hybrid search approach (combining 
    literal keyword matching and semantic context). 
    
    Use this tool whenever you need to find technical details, API specifications, 
    or architectural information regarding the project.

    Args:
        query: The specific question or set of keywords to look for in the docs.

    Returns:
        A list of descriptions and text snippets from the most relevant parts of the repository.
    """
    # Check if we have data to search
    if index is None:
        return "ERROR: The repository index is empty. Please run 'python run_pipeline.py' to generate 'chunks.json' first."
    
    # Use Hybrid search if both indexes are available, otherwise fall back to Text-only
    try:
        if v_index and model:
            results = hybrid_search(query, index, v_index, model, num_results=3)
        else:
            results = index.search(query, num_results=3)
            
        if not results:
            return "No relevant documentation found for that query."
        
        # Format results into a single string for the agent to read
        formatted_results = []
        for i, res in enumerate(results, 1):
            source = res.get('filename', 'Unknown File')
            content = res.get('content') or res.get('section', 'No content')
            # Extract only the first 1000 chars of content for the agent to keep context window clean
            snippet = content[:1000] + "..." if len(content) > 1000 else content
            formatted_results.append(f"--- Source {i}: {source} ---\n{snippet}\n")
            
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error during search execution: {str(e)}"

# --- Execution Entry Point ---

async def run_agent_example():
    """
    Example run of the agent with a sample question.
    """
    question = "What are the API requirements?"
    print(f"\nUser Question: {question}")
    print("-" * 30)
    
    try:
        # Run the agent and wait for the response
        print("Running agent...")
        result = await agent.run(question)
        
        if result is not None:
            print("\nAgent Response:")
            # In this version of pydantic-ai, the result is in .output
            print(result.output)
        else:
            print("\nError: Agent returned None.")
    except Exception as e:
        print(f"\nFailed to run agent: {e}")

if __name__ == "__main__":
    # In an actual application, this would be triggered by a user request
    asyncio.run(run_agent_example())
