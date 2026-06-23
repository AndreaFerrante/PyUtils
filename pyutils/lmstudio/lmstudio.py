# ============================================================================
# HOW TO USE
# ============================================================================

# STEP 1: Get your token
# 1. Open LM Studio
# 2. Go to Developer tab
# 3. Copy the API Token at the top

# STEP 2: Use ONE of these methods:

# METHOD A - Environment variable (RECOMMENDED)
# Run in terminal first:
#   export LM_API_TOKEN="paste-your-token-here"
# Then run:
# response = call_lmstudio("What is AI?")
# print(response)

# METHOD B - Pass token directly
# response = call_lmstudio("What is AI?", api_token="paste-your-token-here")
# print(response)

# METHOD C - Multiple questions
# questions = ["What is Python?", "What is ML?"]
# for q in questions:
#     print(f"Q: {q}")
#     print(f"A: {call_lmstudio(q)}")
#     print()

import requests
import json
import os


def call_lmstudio(prompt: str, api_host: str = "localhost:1234", api_token: str = None) -> str:
    """Call LM Studio with authentication."""
    
    token = api_token or os.getenv("LM_API_TOKEN")
    
    if not token:
        return "ERROR: No API token. Set export LM_API_TOKEN='your-token'"
    
    url = f"http://{api_host}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        data = response.json()
        
        if "error" in data:
            return f"ERROR: {data['error']['message']}"
        
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "output" in data:
            return data["output"]
        else:
            return json.dumps(data)
        
    except Exception as e:
        return f"ERROR: {str(e)}"

