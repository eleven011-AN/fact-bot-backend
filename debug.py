import requests

try:
    resp = requests.post('https://fact-bot-backend.onrender.com/api/verify', json={'type': 'text', 'value': 'The Great Wall of China is visible from space.'})
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print(e)
