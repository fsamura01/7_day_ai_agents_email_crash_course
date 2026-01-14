---
title: Project Scripts
description: Core scripts for fetching and processing GitHub repository documentation.
status: functional
---

# Project Scripts

This directory contains the core scripts for fetching and processing GitHub repository documentation.

## Scripts

### 1. `data_preparation.py`
Fetches markdown files from GitHub repositories.

```python
from data_preparation import read_repo_data

docs = read_repo_data('owner', 'repo-name')
```

### 2. `chunking.py`
Chunks documents using a sliding window approach.

**Run standalone:**
```bash
uv run python chunking.py
```

### 3. `run_pipeline.py` (Recommended)
CLI tool combining both modules.

**Basic usage:**
```bash
uv run python run_pipeline.py <owner> <repo>
```

**With options:**
```bash
uv run python run_pipeline.py fsamura01 task-manager-app \
  --chunk-size 3000 \
  --step-size 1500 \
  --verbose \
  --output chunks.json
```

**Arguments:**
- `--chunk-size`: Characters per chunk (default: 2000)
- `--step-size`: Overlap between chunks (default: 1000)
- `--output`: Save chunks to JSON file
- `--verbose`: Show detailed statistics

## Example Output

```
📥 Fetching data from fsamura01/task-manager-app...
✓ Retrieved 5 documents

🔪 Chunking documents (size=2000, step=1000)...
✓ Created 12 chunks

✅ Pipeline completed successfully!
```
