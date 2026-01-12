import os
import re
import json

def clean_notebook(filepath):
    """Remove API keys from a Jupyter notebook."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        
        # Clean all cells
        for cell in notebook.get('cells', []):
            if 'source' in cell:
                original = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                
                # Replace API keys
                cleaned = re.sub(r'gsk_[a-zA-Z0-9]+', 'REDACTED_GROQ_KEY', original)
                cleaned = re.sub(r'sk-proj-[a-zA-Z0-9_-]+', 'REDACTED_OPENAI_KEY', cleaned)
                cleaned = re.sub(r'sk-[a-zA-Z0-9]+', 'REDACTED_OPENAI_KEY', cleaned)
                
                if cleaned != original:
                    modified = True
                    # Convert back to list format if it was a list
                    if isinstance(cell['source'], list):
                        cell['source'] = cleaned.split('\n')
                        # Preserve the newline structure
                        cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                                        for i, line in enumerate(cell['source'])]
                    else:
                        cell['source'] = cleaned
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            print(f"✓ Cleaned: {filepath}")
            return True
        else:
            print(f"  No secrets found: {filepath}")
            return False
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False

def main():
    """Find and clean all Jupyter notebooks in the repository."""
    cleaned_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in root:
            continue
            
        for file in files:
            if file.endswith('.ipynb'):
                filepath = os.path.join(root, file)
                if clean_notebook(filepath):
                    cleaned_count += 1
    
    print(f"\n{'='*50}")
    print(f"Cleaned {cleaned_count} notebook(s)")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
