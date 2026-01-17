from pydantic_ai import Agent
from search_tools import text_search

SYSTEM_PROMPT = """
You are a technical documentation assistant for a GitHub repository.
Your goal is to answer developer questions specifically using the provided documentation.

RULES:
1. ALWAYS use 'text_search' before answering.
2. If results are missing or the index isn't ready, tell the user to run ingestion.
3. Answer the question based on the search results. If the results are relevant, use them. Only say "Information not found" if the search results satisfy NONE of the user's intent.
4. Be concise and technically accurate.
5. CITATIONS: You will receive search results in the format `[Source: filename](url)`. You MUST include this exact clickable link at the end of your answer.
   - Example Input: `Source 1 [README.md](https://github.com/...)`
   - Example Output: `...answer... [Source: README.md](https://github.com/...)`
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
