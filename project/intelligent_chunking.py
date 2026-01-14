from typing import List, Dict

def setup_llm_client(provider: str = 'groq'):
    """
    Setup LLM client. 
    
    Args:
        provider: 'groq' (free, recommended) or 'openai'
    
    Groq setup:
        1. Get free API key: https://console.groq.com/keys
        2. export GROQ_API_KEY='your-key'
        3. pip install groq
    
    OpenAI setup:
        1. Get API key: https://platform.openai.com/api-keys
        2. export OPENAI_API_KEY='your-key'
        3. pip install openai
    """
    if provider == 'groq':
        try:
            from groq import Groq
            return Groq()
        except ImportError:
            raise ImportError("Install Groq: pip install groq")
        except Exception as e:
            raise Exception(f"Setup failed. Set GROQ_API_KEY: {e}")
    
    elif provider == 'openai':
        try:
            from openai import OpenAI
            return OpenAI()
        except ImportError:
            raise ImportError("Install OpenAI: pip install openai")
        except Exception as e:
            raise Exception(f"Setup failed. Set OPENAI_API_KEY: {e}")
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def llm(client, prompt: str, model: str = None, provider: str = 'groq') -> str:
    """
    Call LLM with prompt and return response.
    
    Args:
        client: LLM client (Groq or OpenAI)
        prompt: Input prompt
        model: Model name (auto-selected if None)
        provider: 'groq' or 'openai'
    
    Returns:
        LLM response text
    """
    # Auto-select model based on provider
    if model is None:
        if provider == 'groq':
            model = 'llama-3.3-70b-versatile'  # Fast and free
        else:
            model = 'gpt-4o-mini'
    
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content


INTELLIGENT_CHUNKING_PROMPT = """
Split the provided document into logical sections that make sense for a Q&A system.

Each section should be:
- Self-contained and cover a specific topic or concept
- Include all relevant context needed to understand the topic
- Between 500-2000 characters when possible

<DOCUMENT>
{document}
</DOCUMENT>

Use this exact format (IMPORTANT: use --- as separator):

## Section Name

Section content with all relevant details

---

## Another Section Name

Another section content

---

Do not add any preamble or explanation, just output the sections.
""".strip()



def intelligent_chunking(client, text: str, provider: str = 'groq') -> List[str]:
    """
    Use LLM to intelligently split document into sections.
    
    Args:
        client: LLM client (Groq or OpenAI)
        text: Document text
        provider: 'groq' or 'openai'
    
    Returns:
        List of section texts
    """
    prompt = INTELLIGENT_CHUNKING_PROMPT.format(document=text)
    response = llm(client, prompt, provider=provider)
    
    # Split by separator
    sections = response.split('---')
    sections = [s.strip() for s in sections if s.strip()]
    
    return sections


def process_documents_intelligent(documents: List[Dict],
                                 client=None,
                                 provider: str = 'groq',
                                 show_progress: bool = True) -> List[Dict]:
    """
    Process documents using LLM-powered intelligent chunking.
    
    FREE option with Groq (recommended)!
    
    Args:
        documents: List of document dicts with 'content' key
        client: LLM client (will create Groq client if None)
        provider: 'groq' (free, default) or 'openai' (costs money)
        show_progress: Show progress bar (requires tqdm)
    
    Returns:
        List of section dictionaries with preserved metadata
    """
    if client is None:
        client = setup_llm_client(provider)
    
    # Safety limit for LLM processing
    MAX_DOC_CHARS_FOR_LLM = 20000
    
    # Optional progress bar
    iterator = documents
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(documents, desc="Processing documents")
        except ImportError:
            print("Install tqdm for progress bars: pip install tqdm")
    
    all_sections = []
    
    for doc_idx, doc in enumerate(iterator):
        # Create copy and extract content
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content', '')
        
        if not doc_content.strip():
            continue

        try:
            # Check for size limit
            if len(doc_content) > MAX_DOC_CHARS_FOR_LLM:
                raise ValueError(f"Content length ({len(doc_content)}) exceeds LLM limit ({MAX_DOC_CHARS_FOR_LLM})")

            # Use LLM to split intelligently
            sections = intelligent_chunking(client, doc_content, provider)
            
            # Add metadata to each section
            for section_idx, section in enumerate(sections):
                chunk_dict = doc_copy.copy()
                chunk_dict['content'] = section
                chunk_dict['content_length'] = len(section)
                chunk_dict['doc_index'] = doc_idx
                chunk_dict['chunk_index'] = section_idx
                chunk_dict['total_chunks'] = len(sections)
                chunk_dict['chunking_method'] = f'intelligent_{provider}'
                all_sections.append(chunk_dict)
        
        except Exception as e:
            print(f"Error processing doc {doc_idx} ({doc_copy.get('filename', 'unknown')}): {e}")
            
            # Fallback to simple chunking on error
            from chunking import sliding_window
            # Fallback uses standard 2000/1000 params
            chunks = sliding_window(doc_content, 2000, 1000)
            for chunk in chunks:
                chunk_dict = chunk.copy()
                chunk_dict.update(doc_copy)
                # Ensure fallback also uses 'content' key if sliding_window returns 'chunk'
                if 'chunk' in chunk_dict:
                    chunk_dict['content'] = chunk_dict.pop('chunk')
                chunk_dict['chunking_method'] = 'fallback_simple'
                all_sections.append(chunk_dict)
    
    return all_sections