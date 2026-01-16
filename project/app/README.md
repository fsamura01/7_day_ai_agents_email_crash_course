# 🤖 Repository Documentation Agent

A modular, AI-powered documentation assistant that can ingest any GitHub repository and answer questions about it using hybrid search (Keyword + Semantic).

## ✨ Features

- **Hybrid Search**: Combines TF-IDF keyword matching with `multi-qa-distilbert-cos-v1` vector embeddings for high accuracy.
- **Real-time Streaming**: Responses are streamed token-by-token for a fluid user experience.
- **Conversation History**: The agent remembers previous turns, allowing for follow-up questions.
- **Auto-Ingestion**: Fetches, cleans, chunks, and indexes markdown files from any public GitHub repo.
- **Self-Correcting**: Handles missing dependencies or API limits gracefully.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Groq API Key](https://console.groq.com/keys)

### Installation

1. **Navigate to the app directory**:
   ```bash
   cd project/app
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```
   *Or with pip:* `pip install -r requirements.txt`

3. **Configure Environment**:
   Create a `.env` file in this directory:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

### Running the App

Start the Streamlit interface:
```bash
uv run streamlit run main.py
```

## 📂 Architecture

- **`main.py`**: The Streamlit entry point. Handles UI, session state, and the async event loop for streaming.
- **`ingest.py`**: ETL pipeline. Downloads repo zip, parses markdown, chunks content, and generates embeddings.
- **`search_tools.py`**: The "brain" of the search. Manages TF-IDF and Vector indexes and performs the hybrid reranking.
- **`search_agent.py`**: Defines the Pydantic AI Agent, including system prompts and tool registration.
- **`logs.py`**: Structured JSON logging for all agent interactions.

## 🛠️ Configuration

**Changing the Repository**:
You don't need to change code! Just use the **Sidebar** in the app to enter a new `Owner` and `Repo Name`, then click **Ingest Repository**.
