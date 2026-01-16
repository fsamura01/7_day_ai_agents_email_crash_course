from search_tools import initialize_search_indexes, text_search
import os

def debug_search():
    print("Initializing indexes...")
    success = initialize_search_indexes()
    if not success:
        print("Failed to initialize indexes. Check chunks.json")
        return

    queries = [
        "How do I install the server?",
        "install dependencies",
        "npm start"
    ]

    for q in queries:
        print(f"\nQUERY: '{q}'")
        result = text_search(q)
        print("-" * 40)
        print(result)
        print("-" * 40)

if __name__ == "__main__":
    debug_search()
