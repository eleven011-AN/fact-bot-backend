import requests  # type: ignore
import json
import time

API_URL = "https://fact-bot-backend.onrender.com/api/verify"

complex_text = "The Great Wall of China is the only man-made object visible from space with the naked eye. Also, water boils at 100 degrees Celsius at standard atmospheric pressure. Finally, eating raw chicken is considered a highly recommended healthy diet for human cardiovascular health."

print(f"Testing Pipeline with complex text...")
payload = {'type': 'text', 'value': complex_text}
start = time.time()
try:
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    print(f'Time: {time.time() - start:.2f}s')
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f'Error: {e}')
