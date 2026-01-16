from pydantic_ai import Agent
from search_tools import text_search

SYSTEM_PROMPT = """
You are a technical documentation assistant for a GitHub repository.
Your goal is to answer developer questions specifically using the provided documentation.

RULES:
1. ALWAYS use 'text_search' before answering.
2. If results are missing or the index isn't ready, tell the user to run ingestion.
3. If the answer isn't in the docs, say "Information not found in current documentation."
4. Be concise and technically accurate.
"""

def create_agent(model_name='groq:llama-3.1-8b-instant'):
    """
    Factory function to create and configure the documentation agent.
    """
    agent = Agent(
        model_name,
        name="docs_agent",
        system_prompt=SYSTEM_PROMPT,
    )
    
    # Register the search tool
    agent.tool_plain(text_search)
    
    return agent
