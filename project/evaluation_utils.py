import json
import secrets
from pathlib import Path
from datetime import datetime
from pydantic_ai.messages import ModelMessagesTypeAdapter

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Fallback for other non-serializable types (like methods or complex objects)
    return str(obj)

def log_entry(agent, messages, source="user"):
    """
    Extracts agent configuration and conversation history into a serializable dictionary.
    """
    # In Pydantic AI, we can get tool names from the agent
    tools = list(agent._function_tools.keys()) if hasattr(agent, '_function_tools') else ['text_search']

    # Convert Pydantic AI messages to plain dictionaries
    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name or "repo_documentation_agent",
        "system_prompt": agent.system_prompt,
        "model": str(agent.model),
        "tools": tools,
        "messages": dict_messages,
        "source": source
    }

def log_interaction_to_file(agent, messages, source='user'):
    """
    Saves the interaction log to a JSON file in the logs directory.
    """
    entry = log_entry(agent, messages, source)

    # Extract the timestamp from the last message for the filename
    # Handle different formats of timestamps (string vs datetime)
    last_msg = entry['messages'][-1]
    ts = last_msg.get('timestamp')
    
    if not ts:
        ts_obj = datetime.now()
    elif isinstance(ts, datetime):
        ts_obj = ts
    else:
        try:
            # Handle ISO format with Z or UTC offset
            ts_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts_obj = datetime.now()

    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    agent_name = agent.name or "agent"
    filename = f"{agent_name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    print(f"[OK] Interaction logged to {filepath}")
    return str(filepath)
