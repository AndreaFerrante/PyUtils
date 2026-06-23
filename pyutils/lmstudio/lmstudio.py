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
import base64
import json
import os


def _prepare_image(image_input):
    """Convert image to base64 or URL."""
    
    # URL
    if isinstance(image_input, str) and image_input.startswith("http"):
        return image_input
    
    # File path
    if os.path.exists(image_input):
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    # PIL Image
    from PIL import Image
    if isinstance(image_input, Image.Image):
        import io
        buffer = io.BytesIO()
        image_input.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

def call_lmstudio_with_image(prompt: str, image_input=None, api_host: str = "localhost:1234", api_token: str = None) -> str:
    """Call LM Studio with IMAGE support."""
    
    token = api_token or os.getenv("LM_API_TOKEN")
    if not token:
        return "ERROR: No API token"
    
    # Prepare image
    image_data = _prepare_image(image_input) if image_input else None
    
    # Build message with text + image
    message_content = [{"type": "text", "text": prompt}]
    
    if image_data:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": image_data if image_data.startswith(("http", "data:")) else f"data:image/jpeg;base64,{image_data}"}
        })
    
    url = f"http://{api_host}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": message_content}],
        "stream": False,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    data = response.json()
    
    return data["choices"][0]["message"]["content"]
