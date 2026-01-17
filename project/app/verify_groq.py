from dotenv import load_dotenv
import os
import requests

load_dotenv(override=True)

api_key = os.environ.get("GROQ_API_KEY")

print(f"Loaded Key Length: {len(api_key) if api_key else 'None'}")
if api_key:
    print(f"Key Prefix: {api_key[:10]}...")

url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    print(f"\nTesting connection to {url}...")
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
    
    if response.status_code == 200:
        print("\n✅ API Key is VALID and working!")
    elif response.status_code == 401:
        print("\n❌ API Key is INVALID (401 Unauthorized).")
    else:
        print(f"\n⚠️ Unexpected error: {response.status_code}")
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
