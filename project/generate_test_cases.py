import json
import random
import asyncio
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Configuration
CHUNK_FILE = "chunks.json"
OUTPUT_FILE = "test_cases.json"
SAMPLE_SIZE = 15

class QuestionList(BaseModel):
    questions: List[str] = Field(description="A list of 1-2 realistic questions based on the content.")

# Initialize the generator agent
# Using Llama 3.1 8B for fast generation
generator = Agent(
    'groq:llama-3.1-8b-instant',
    name="question_generator",
    system_prompt="""
    You are an expert technical documentation reviewer.
    Your task is to generate realistic, specific questions based on snippets of repository documentation.
    
    RULES:
    1. Generate exactly 1-2 questions per content snippet.
    2. Focus on technical details (how to, where is, what are the requirements).
    3. Ensure the question can be answered specifically by the snippet provided.
    4. Keep the questions conversational but professional.
    """,
    output_type=QuestionList
)

async def main():
    if not Path(CHUNK_FILE).exists():
        print(f"Error: {CHUNK_FILE} not found. Run the pipeline first.")
        return

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")
    
    # Sample chunks to create a varied test set
    sample = random.sample(chunks, min(SAMPLE_SIZE, len(chunks)))
    
    test_cases = []
    
    print(f"Generating questions for {len(sample)} chunks...")
    
    for i, chunk in enumerate(sample, 1):
        content = chunk.get('content', '')
        if not content:
            continue
            
        try:
            # We use result_type, and in your environment, result.data or result.output contains the result.
            # Based on previous troubleshooting, we'll try to access .output if it's there.
            result = await generator.run(content[:2000])
            
            # Use the structured data from result.data if it exists (standard Pydantic AI), 
            # otherwise fallback to result.output
            if hasattr(result, 'data'):
                generated_questions = result.data.questions
            else:
                generated_questions = result.output.questions
            
            for q in generated_questions:
                test_cases.append({
                    "id": f"test_{len(test_cases)+1}",
                    "question": q,
                    "expected_file": chunk.get('filename', 'unknown'),
                    "context_snippet": content[:200] + "..." 
                })
            
            print(f"[{i}/{len(sample)}] Generated {len(generated_questions)} questions.")
        except Exception as e:
            print(f"Error generating for chunk {i}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)
        
    print(f"\n[OK] Created {len(test_cases)} test cases in {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
