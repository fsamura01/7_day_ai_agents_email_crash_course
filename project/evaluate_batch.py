import json
import asyncio
from agent import agent
from evaluation_utils import log_interaction_to_file
from judge import evaluate_log

# Configuration
TEST_CASES_FILE = "test_cases.json"
RESULTS_FILE = "evaluation_results.json"
SAMPLE_LIMIT = 2 # Set to a low number for initial testing

async def run_batch_evaluation():
    """
    Orchestrates the evaluation of the agent against synthetic test cases.
    """
    try:
        with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: {TEST_CASES_FILE} not found. Run generate_test_cases.py first.")
        return

    # Take a sample of test cases
    sample = test_cases[:SAMPLE_LIMIT]
    
    print(f"Starting batch evaluation for {len(sample)} test cases...")
    
    results = []
    
    for i, tc in enumerate(sample, 1):
        question = tc['question']
        print(f"\n--- Test {i}/{len(sample)} ---")
        print(f"Question: {question}")
        
        try:
            # 1. Run the agent to get a response
            print("Running agent...")
            agent_result = await agent.run(question)
            
            # 2. Log the full conversation interaction
            # We use result.new_messages() to get the messages from the latest run
            log_path = log_interaction_to_file(agent, agent_result.new_messages())
            
            # 3. Evaluate the logged interaction using the LLM judge
            evaluation = await evaluate_log(log_path)
            
            # 4. Store the results (convert Pydantic models to dicts)
            eval_dict = evaluation.dict() if hasattr(evaluation, 'dict') else (
                evaluation.model_dump() if hasattr(evaluation, 'model_dump') else evaluation
            )
            
            results.append({
                "test_case_id": tc.get('id'),
                "question": question,
                "agent_response": agent_result.output,
                "evaluation": eval_dict,
                "log_path": log_path
            })
            
            # Print a quick summary of the judge's verdict
            pass_count = sum(1 for check in eval_dict.get('checklist', []) if check.get('check_pass'))
            total_checks = len(eval_dict.get('checklist', []))
            print(f"Judge Verdict: {pass_count}/{total_checks} checks passed.")
            print(f"Summary: {eval_dict.get('summary')[:100]}...")
            
        except Exception as e:
            print(f"Error during test case {i}: {str(e)}")

    # Custom serializer to handle Pydantic models and other types
    def custom_serializer(obj):
        if hasattr(obj, 'dict'):
            return obj.dict()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return str(obj)

    # Save final results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=custom_serializer)
    
    print(f"\n[OK] Evaluation complete. Detailed results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    asyncio.run(run_batch_evaluation())
