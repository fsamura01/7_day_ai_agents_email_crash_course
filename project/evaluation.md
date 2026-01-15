# Day 5 Evaluation Implementation Detail

This document provides a technical breakdown of the evaluation framework implemented on Day 5 to verify the performance and accuracy of the Repository Documentation Agent.

## 1. Interaction Logging (`evaluation_utils.py`)
The foundation of our evaluation is the ability to capture and serialize agent interactions.

- **`log_interaction_to_file`**: Captures the full Pydantic AI `AgentRunResult` and its message history.
- **Serialization Handling**: Includes a custom JSON serializer to handle `datetime` objects and "redact" or handle non-serializable Pydantic objects (like bound methods) that often appear in complex AI logs.
- **Storage**: Logs are saved in the `logs/` directory with unique timestamps and hex hashes: `repo_documentation_agent_YYYYMMDD_HHMMSS_random.json`.

## 2. Synthetic Test Case Generation (`generate_test_cases.py`)
To evaluate the agent without manual labeling, we use an LLM to "bootstrap" a dataset from the existing documentation.

- **Model**: `groq:llama-3.1-8b-instant`.
- **Logic**: 
  - Loads `chunks.json`.
  - Samples random documentation snippets.
  - Prompts the LLM to generate 1-2 specific, technical questions that can be answered *solely* by that snippet.
- **Output**: Generates `test_cases.json` containing the question, the original source filename, and a context snippet for reference.

## 3. The LLM Judge (`judge.py`)
We use a high-reasoning model to grade the agent's performance based on structured criteria.

- **Model**: `groq:llama-3.3-70b-versatile`.
- **Structured Schema**: Uses Pydantic models to ensure the judge returns a consistent `EvaluationChecklist`.
- **Criteria**:
  - `instructions_follow`: Compliance with system prompt rules.
  - `answer_relevant`: Directness of the response.
  - `answer_clear`: Clarity and technical accuracy.
  - `answer_citations`: Verification of source attribution.
  - `completeness`: Ensuring no part of the query was ignored.
  - `tool_call_search`: Verification that the search tool was actually utilized.
- **Log Simplification**: To save tokens and focus the judge, the script redacts large technical search results from the conversation history, keeping only the fact that the tool was called and returned data.

## 4. Batch Orchestration (`evaluate_batch.py`)
This script automates the full evaluation cycle.

1. Loads the generated test cases.
2. Runs theDocumentation Agent for each question.
3. Logs the interaction to a JSON file.
4. Invokes the LLM Judge to grade that specific log.
5. Aggregates results into `evaluation_results.json`.

## 5. Interactive Review (`evaluate.ipynb`)
A Jupyter notebook designed for human review of the metrics.

- Loads `evaluation_results.json`.
- Displays a **Results Summary Board** using Pandas, with status icons (✅, ❌, ➖).
- Provides a **Deep Dive** section that shows the raw agent response alongside the judge's justification for every check.

## 🛠 Usage Flow
```bash
# 1. Generate questions
python generate_test_cases.py

# 2. Run agent and grade results
python evaluate_batch.py

# 3. View dashboard
# Open evaluate.ipynb in Jupyter
```
