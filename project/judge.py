import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# --- Structured Output Models ---

class EvaluationCheck(BaseModel):
    check_name: str = Field(description="Name of the check being performed (e.g., answer_relevant)")
    justification: str = Field(description="Short explanation for why the check passed or failed")
    check_pass: Optional[bool] = Field(description="True if the criteria was met, False otherwise")

class EvaluationChecklist(BaseModel):
    checklist: List[EvaluationCheck] = Field(description="List of specific quality checks")
    summary: str = Field(description="Overall verdict on the agent's response")

# --- Evaluation Logic ---

EVALUATION_PROMPT = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).
We also include the entire log (<LOG>) for analysis.

For each item, check if the condition is met. 

Checklist:
- instructions_follow: Did the agent follow its system instructions?
- answer_relevant: Does the response directly answer the user's specific question?
- answer_clear: Is the answer easy to understand and technically accurate?
- answer_citations: Does the response cite the source file correctly if available?
- completeness: Does the answer cover all parts of the user's inquiry?
- tool_call_search: Did the agent actually use the 'text_search' tool to find facts?

You MUST output your evaluation in the structured format requested.
""".strip()

# Judge Agent: Using Llama 3.3 70B for high-quality reasoning
eval_agent = Agent(
    'groq:llama-3.3-70b-versatile',
    name='eval_agent',
    system_prompt=EVALUATION_PROMPT,
    output_type=EvaluationChecklist
)

def simplify_log_messages(messages):
    """
    Strips unnecessary metadata and redacts large tool returns to focus the judge
    on the reasoning and final output while saving tokens.
    """
    log_simplified = []
    for m in messages:
        parts = []
        for original_part in m.get('parts', []):
            part = original_part.copy()
            kind = part.get('part_kind')
            
            # Remove bulky metadata standard in Pydantic AI logs
            keys_to_remove = ['timestamp', 'tool_call_id', 'metadata', 'id']
            for key in keys_to_remove:
                if key in part: del part[key]
            
            # Redact tool results to keep the prompt size manageable
            if kind == 'tool-return':
                # We just need to know the tool was called and returned something
                part['content'] = '[TECHNICAL_RESULTS_REDACTED_FOR_BREVITY]'
            
            parts.append(part)
        
        log_simplified.append({
            'kind': m.get('kind'),
            'parts': parts
        })
    return log_simplified

async def evaluate_log(log_path: str):
    """
    Loads a JSON log file and runs the evaluation agent against it.
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        log_record = json.load(f)
        
    messages = log_record['messages']
    instructions = log_record['system_prompt']
    
    # In Pydantic AI loops, first part of first message is usually the question
    question = messages[0]['parts'][0]['content']
    # Last part of last message is the response
    answer = messages[-1]['parts'][0]['content']
    
    log_simplified = simplify_log_messages(messages)
    
    user_prompt = f"""
    <INSTRUCTIONS>{instructions}</INSTRUCTIONS>
    <QUESTION>{question}</QUESTION>
    <ANSWER>{answer}</ANSWER>
    <LOG>{json.dumps(log_simplified)}</LOG>
    """.strip()
    
    print(f"Evaluating log: {log_path}...")
    result = await eval_agent.run(user_prompt)
    
    # Handle version differences in Pydantic AI result retrieval
    eval_data = result.data if hasattr(result, 'data') else result.output
    return eval_data

if __name__ == "__main__":
    import sys
    import os
    
    async def cli_run():
        if len(sys.argv) < 2:
            print("Usage: python judge.py <path_to_log_json>")
            return
            
        log_file = sys.argv[1]
        if not os.path.exists(log_file):
            print(f"Error: File {log_file} not found.")
            return
            
        result = await evaluate_log(log_file)
        
        print("\n--- Evaluation Summary ---")
        print(result.summary)
        print("\n--- Detailed Checks ---")
        for check in result.checklist:
            status = "[PASS]" if check.check_pass else "[FAIL]"
            print(f"{status} {check.check_name}: {check.justification}")

    asyncio.run(cli_run())
