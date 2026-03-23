import requests

try:
    resp = requests.post('http://localhost:8000/api/verify', json={'type': 'text', 'value': 'The Great Wall of China is visible from space.'})
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print(e)
