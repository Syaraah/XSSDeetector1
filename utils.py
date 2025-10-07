import pandas as pd
import html
import urllib.parse
import re
import logging
from typing import List, Dict, Tuple, Optional

# Setup logging
logger = logging.getLogger(__name__)

# === XSS PATTERNS & VALIDATION ===
XSS_PATTERNS = {
    'script_tags': [
        r'<script[^>]*>.*?</script>',
        r'<script[^>]*>',
        r'</script>'
    ],
    'event_handlers': [
        r'on\w+\s*=',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
        r'onmouseover\s*=',
        r'onfocus\s*=',
        r'onblur\s*='
    ],
    'javascript_protocols': [
        r'javascript:',
        r'vbscript:',
        r'data:text/html',
        r'data:application/x-javascript'
    ],
    'dangerous_tags': [
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'<form[^>]*>',
        r'<input[^>]*>',
        r'<textarea[^>]*>',
        r'<select[^>]*>'
    ],
    'encoding_evasion': [
        r'%3Cscript%3E',
        r'%3C/script%3E',
        r'&#x3C;script&#x3E;',
        r'&#60;script&#62;',
        r'\\x3Cscript\\x3E',
        r'\\u003Cscript\\u003E'
    ]
}

# === DATA PREPROCESSING (CLEANING, FILTERING, AUGMENTASI) ===
def clean_payload(payload: str) -> str:
    """Clean and normalize payload for analysis"""
    try:
        if not payload or not isinstance(payload, str):
            return ""
        
        # HTML decode
        payload = html.unescape(payload)
        
        # URL decode (multiple times to handle nested encoding)
        for _ in range(3):
            try:
                payload = urllib.parse.unquote(payload)
            except:
                break
        
        # Convert to lowercase
        payload = payload.lower()
        
        # Replace numbers with 0
        payload = re.sub(r'\d+', '0', payload)
        
        # Replace URLs
        payload = re.sub(r'(http|https)://[^\s]+', "http://u", payload)
        
        # Remove extra whitespace
        payload = re.sub(r'\s+', ' ', payload).strip()
        
        return payload
    except Exception as e:
        logger.error(f"Error cleaning payload: {e}")
        return str(payload) if payload else ""

def is_valid_payload(payload: str) -> bool:
    """Check if payload contains potential XSS patterns"""
    try:
        if not payload:
            return False
        
        payload_lower = payload.lower()
        
        # Check for basic XSS indicators
        basic_indicators = ['<', '>', 'script', 'onerror', 'onload', 'img', 'alert', 'javascript']
        if any(indicator in payload_lower for indicator in basic_indicators):
            return True
        
        # Check for encoded patterns
        for pattern_type, patterns in XSS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, payload_lower, re.IGNORECASE):
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error validating payload: {e}")
        return False

def detect_xss_patterns(payload: str) -> Dict[str, List[str]]:
    """Detect specific XSS patterns in payload"""
    detected_patterns = {}
    
    try:
        payload_lower = payload.lower()
        
        for pattern_type, patterns in XSS_PATTERNS.items():
            found_patterns = []
            for pattern in patterns:
                matches = re.findall(pattern, payload_lower, re.IGNORECASE)
                if matches:
                    found_patterns.extend(matches)
            
            if found_patterns:
                detected_patterns[pattern_type] = list(set(found_patterns))
        
        return detected_patterns
    except Exception as e:
        logger.error(f"Error detecting XSS patterns: {e}")
        return {}

def augment_payload(payload: str) -> List[str]:
    """Generate payload variations for training"""
    variations = set()
    
    try:
        if not payload:
            return []
        
        variations.add(payload)
        
        # URL encoding variations
        variations.add(payload.replace("<", "%3C"))
        variations.add(payload.replace(">", "%3E"))
        variations.add(payload.replace("script", "%73cript"))
        
        # Case variations
        variations.add(payload.replace("script", "ScRiPt"))
        variations.add(payload.replace("alert", "AlErT"))
        
        # HTML comment injection
        variations.add(payload.replace("<script>", "<scr<!--x-->ipt>"))
        variations.add(payload.replace("</script>", "</scr<!--x-->ipt>"))
        
        # Double encoding
        variations.add(payload.replace("<", "%253C"))
        variations.add(payload.replace(">", "%253E"))
        
        # Unicode encoding
        variations.add(payload.replace("<", "\\u003C"))
        variations.add(payload.replace(">", "\\u003E"))
        
        return list(variations)
    except Exception as e:
        logger.error(f"Error augmenting payload: {e}")
        return [payload] if payload else []

def sanitize_output(text: str) -> str:
    """Sanitize text for safe output"""
    try:
        if not text:
            return ""
        
        # HTML escape
        text = html.escape(text)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', '', text)
        
        return text
    except Exception as e:
        logger.error(f"Error sanitizing output: {e}")
        return str(text) if text else ""

def validate_input_length(payload: str, max_length: int = 10000) -> Tuple[bool, str]:
    """Validate input length"""
    if not payload:
        return False, "Payload is empty"
    
    if len(payload) > max_length:
        return False, f"Payload too long (max {max_length} characters)"
    
    return True, "Valid length"

def extract_urls_from_payload(payload: str) -> List[str]:
    """Extract URLs from payload for analysis"""
    try:
        url_pattern = r'https?://[^\s<>"\']+'
        urls = re.findall(url_pattern, payload, re.IGNORECASE)
        return list(set(urls))
    except Exception as e:
        logger.error(f"Error extracting URLs: {e}")
        return []

def preprocess_dataset(input_csv='data/XSS_dataset.csv', output_csv='data/XSS_dataset_cleaned.csv', augment=True):
    """Preprocess dataset for training"""
    try:
        df = pd.read_csv(input_csv)

        # Validate columns
        if 'Payload' not in df.columns or 'Label' not in df.columns:
            raise ValueError("Dataset harus memiliki kolom 'Payload' dan 'Label'")

        # Cleaning dan filtering
        df['Payload'] = df['Payload'].astype(str)
        df['Cleaned'] = df['Payload'].apply(clean_payload)
        df = df[df['Cleaned'].apply(is_valid_payload)].copy()

        # Augmentasi
        if augment:
            augmented = []
            for _, row in df[df['Label'] == 1].iterrows():
                variants = augment_payload(row['Cleaned'])
                for variant in variants:
                    augmented.append({
                        'Payload': variant, 
                        'Label': 1, 
                        'Cleaned': clean_payload(variant)
                    })
            df_aug = pd.DataFrame(augmented)
            df = pd.concat([df, df_aug], ignore_index=True)

        # Simpan
        df[['Payload', 'Label', 'Cleaned']].to_csv(output_csv, index=False)
        print(f"✅ Dataset berhasil disimpan ke {output_csv} ({len(df)} baris)")
        
        return df
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise

def analyze_payload_security(payload: str) -> Dict[str, any]:
    """Comprehensive payload security analysis"""
    try:
        cleaned = clean_payload(payload)
        patterns = detect_xss_patterns(payload)
        urls = extract_urls_from_payload(payload)
        
        return {
            "original_length": len(payload),
            "cleaned_length": len(cleaned),
            "contains_xss_patterns": len(patterns) > 0,
            "detected_patterns": patterns,
            "urls_found": urls,
            "has_encoding": any(char in payload for char in ['%', '\\', '&#']),
            "has_events": any('on' in pattern for patterns in patterns.values() for pattern in patterns),
            "risk_score": calculate_risk_score(payload, patterns)
        }
    except Exception as e:
        logger.error(f"Error analyzing payload security: {e}")
        return {}

def calculate_risk_score(payload: str, patterns: Dict[str, List[str]]) -> float:
    """Calculate risk score based on payload analysis"""
    try:
        score = 0.0
        
        # Base score for having patterns
        if patterns:
            score += 0.3
        
        # Pattern-specific scoring
        if 'script_tags' in patterns:
            score += 0.4
        if 'event_handlers' in patterns:
            score += 0.3
        if 'javascript_protocols' in patterns:
            score += 0.5
        if 'dangerous_tags' in patterns:
            score += 0.2
        if 'encoding_evasion' in patterns:
            score += 0.3
        
        # Length factor
        if len(payload) > 1000:
            score += 0.1
        
        return min(score, 1.0)
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        return 0.0

if __name__ == '__main__':
    preprocess_dataset()



