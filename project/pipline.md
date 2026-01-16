# Pipeline Execution Flow

This document tracks how data flows through `run_pipeline.py` and where the key functions are called.

## Detailed Pipeline Execution

Here is where the major functions are called in `project/run_pipeline.py`.

### Step 1: Fetching (Line 62)
Function: read_repo_data
Location: Defined in ingest.py
Purpose: Connects to GitHub, downloads the repository, and extracts raw text from all Markdown files.

### Step 2: Chunking (Line 70)
Function: chunk_documents
Location: Defined in chunking.py
Purpose: Takes the large documents from Step 1 and splits them into smaller, manageable sections.

### Step 3: Intelligent Refinement
Function: process_documents_intelligent (Called inside chunk_documents)
Location: Defined in intelligent_chunking.py
Purpose: Uses AI to find logical section breaks instead of just cutting by character count.

### Step 4: Output Saving (Line 102)
Action: json.dump
Result: The final list of processed chunks is stored permanently in chunks.json.

### Step 5: Keyword Indexing (Line 118)
Action: index.fit
Location: search.py
Purpose: Scans all chunks to build a mathematical Keyword Map for fast text searching.

### Step 6: Semantic Indexing (Line 147)
Action: v_index.fit
Location: search.py
Purpose: Loads the AI-generated vectors into the search engine for meaning-based lookups.

### Step 7: Search Loop (Line 166)
Function: hybrid_search
Location: search.py
Purpose: Found in the Interactive Prompt. It combines Keyword and Semantic indexes to answer queries.

### Stage 8: Cache Validation and Auto-Correction
Action: Automatic mismatch detection
Purpose: If you change your documents, the number of chunks will no longer match the saved vectors in embeddings.npy.
How it works: The script detects this mismatch (Line 137), deletes the outdated index, and automatically re-computes the semantic vectors so your search results are always accurate.

---

## Understanding the Command Line Options
When you run python run_pipeline.py --help, you see the instruction manual for the tool.

### Required Information (Positional Arguments)
repo_owner: The GitHub username or organization (e.g., fsamura01).
repo_name: The name of the project repository (e.g., task-manager-app).

### Customization Options
--chunk-size: How many characters per text piece (default is 2000).
--step-size: The overlap between pieces to ensure no text is lost (default is 1000).
--output: The filename if you want to save the results (e.g., chunks.json).
--verbose: Show detailed statistics and a preview of the data on your screen.
--interactive: Start the AI Search prompt as soon as the processing finishes.

---

## How to use this map
1. Open project/run_pipeline.py.
2. Go to the Line numbers mentioned in the headers above.
3. You will see the exact point where data is handed off to the next module.