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
    return str(obj)

def log_interaction(agent, messages, source='user'):
    """
    Saves the interaction log to a JSON file in the logs directory.
    """
    # Tools extraction
    tools = list(agent._function_tools.keys()) if hasattr(agent, '_function_tools') else []

    # Convert Pydantic AI messages to plain dictionaries
    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    entry = {
        "agent_name": agent.name or "agent",
        "system_prompt": agent.system_prompt,
        "model": str(agent.model),
        "tools": tools,
        "messages": dict_messages,
        "source": source
    }

    # Timestamp for filename
    last_msg = entry['messages'][-1]
    ts = last_msg.get('timestamp')
    
    if not ts:
        ts_obj = datetime.now()
    elif isinstance(ts, datetime):
        ts_obj = ts
    else:
        try:
            ts_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            ts_obj = datetime.now()

    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)
    filename = f"{entry['agent_name']}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return str(filepath)
