import re
import urllib.parse
import html

def clean_payload(payload: str) -> str:
    """
    Membersihkan dan menormalkan payload dari XSS attack:
    - Decode HTML entities
    - Decode URL encoding
    - Hilangkan tag HTML
    - Hilangkan karakter tidak penting
    """
    if not isinstance(payload, str):
        return ""

    payload = html.unescape(payload) # 1. HTML decode
    payload = urllib.parse.unquote(payload) # 2. URL decode
    payload = payload.lower()  # 3. Lowercase
    payload = re.sub(r'<.*?>', '', payload) # 4. Hapus tag HTML
    payload = re.sub(r'[^a-z0-9<>/"\'=;:_-]', ' ', payload)  
    payload = re.sub(r'\s+', ' ', payload).strip()
    return payload
