import re
import urllib.parse
import html

def clean_payload(payload: str) -> str:
    if not isinstance(payload, str):
        return ""

    payload = html.unescape(payload) #  HTML decode
    payload = urllib.parse.unquote(payload) # URL decode
    payload = payload.lower()  # Lowercase
    payload = re.sub(r'[^a-z0-9<>/"\'=;:_-]', ' ', payload)  
    payload = re.sub(r'\s+', ' ', payload).strip()
    return payload
