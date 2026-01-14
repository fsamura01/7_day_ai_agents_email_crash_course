# Repo AI Agent Guide

This document explains the architecture and logic of the search agent implemented in agent.py using Pydantic AI.

## Architecture Overview
The agent acts as a reasoning layer on top of your existing search indexes.

### Basic Steps
The agent has access to a text_search tool. When you ask a question, the agent decides if it needs to search the repository. It then uses the retrieved information to provide a grounded, accurate answer instead of hallucinating.

---

## Logic Breakdown

### 1. Hybrid Search Tool
The text_search function is the core tool. The agent generates a search query which is passed to the hybrid_search logic in search.py. This retrieves the best matches from both keyword (Text) and semantic (Vector) indexes. The output is a formatted string containing the top 3 most relevant documentation chunks.

### 2. Caching Mechanism
The agent is designed to be efficient. On startup, it looks for chunks.json and embeddings.npy. 

If found (Cache Hit), it loads the indexes into memory instantly. 

If missing (Cache Miss), the tool returns a message informing the agent that documentation is not loaded. The agent then informs the user to run the pipeline.

### 3. Systematic Reasoning
The agent uses a System Prompt to define its persona. It knows it is a technical documentation assistant and is instructed to only answer based on the search results. If it cannot find an answer in the search results, it truthfully admits it does not know.

---

## How to Use

### Prerequisites
Ensure you have processed a repository first using this command:
python run_pipeline.py owner repo --output chunks.json

### Running the Agent
Execute the agent script directly:
python agent.py

---

## Actions and Workflow
Step 1: Initialize Indexes. This loads text and vector data from disk.
Step 2: User Input. The agent awaits a natural language question.
Step 3: Reasoning. The LLM analyzes the question and generates a search query.
Step 4: Execution. The text_search tool is triggered to find facts.
Step 5: Synthesis. The LLM combines the search results into a clean, human-friendly response.

---

## Troubleshooting and Fixes

During the initial setup, several platform-specific and version-specific issues were resolved.

### 1. Pydantic AI Initialization Error
Error: UserError: Unknown keyword argument description.
Fix: The Agent class used the name parameter instead of description to comply with the library version.

### 2. Attribute Error in Run Result
Error: AttributeError: AgentRunResult object has no attribute data.
Fix: The response logic was updated to use result.output instead of result.data to correctly retrieve the agent's text response.

### 3. Windows Terminal Encoding Crash
Error: charmap codec cannot encode character u2713.
Fix: Unicode characters like checkmarks were replaced with plain text markers like [OK] to prevent crashes on Windows terminals.
