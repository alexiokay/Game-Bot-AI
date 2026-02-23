import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

print("Listing available models...")
try:
    for m in client.models.list(config={"page_size": 100}):
        # Just print the name based on typical SDK response
        print(f"- {m.name if hasattr(m, 'name') else m}")
except Exception as e:
    print(f"Error: {e}")
