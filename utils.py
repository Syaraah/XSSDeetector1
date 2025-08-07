import pandas as pd
import html
import urllib.parse
import re

# === DATA PREPROCESSING (CLEANING, FILTERING, AUGMENTASI) ===
def clean_payload(payload: str) -> str:
    payload = html.unescape(payload)  # HTML decode
    payload = urllib.parse.unquote(payload)  # URL decode
    payload = payload.lower()  # Lowercase
    payload = re.sub(r'\d+', '0', payload)  # Ganti angka dengan 0
    payload = re.sub(r'(http|https)://[^\s]+', "http://u", payload)  # Ganti URL
    payload = re.sub(r'\s+', ' ', payload).strip()  # Hilangkan spasi berlebih
    return payload

def is_valid_payload(payload: str) -> bool:
    tags = ['<', '>', 'script', 'onerror', 'onload', 'img', 'alert']
    return any(tag in payload for tag in tags)

def augment_payload(payload: str) -> list:
    variations = set()
    variations.add(payload)  
    variations.add(payload.replace("<", "%3C"))  # URL encoding
    variations.add(payload.replace("script", "ScRiPt"))  # Case variation
    variations.add(payload.replace("<script>", "<scr<!--x-->ipt>"))  # HTML comment injection
    return variations

def preprocess_dataset(input_csv='data/XSS_dataset.csv', output_csv='data/XSS_dataset_cleaned.csv', augment=True):
    df = pd.read_csv(input_csv)

    # Validasi kolom
    if 'Payload' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'Payload' dan 'Label'")

    # Cleaning dan filtering
    df['Payload'] = df['Payload'].astype(str)
    df['Cleaned'] = df['Payload'].apply(clean_payload)
    df = df[df['Cleaned'].apply(is_valid_payload)].copy()

    #
    if augment:
        augmented = []
        for _, row in df[df['Label'] == 1].iterrows():
            variants = augment_payload(row['Cleaned'])
            for variant in variants:
                augmented.append({'Payload': variant, 'Label': 1, 'Cleaned': clean_payload(variant)})
        df_aug = pd.DataFrame(augmented)
        df = pd.concat([df, df_aug], ignore_index=True)

    # Simpan
    df[['Payload', 'Label', 'Cleaned']].to_csv(output_csv, index=False)
    print(f"✅ Dataset berhasil disimpan ke {output_csv} ({len(df)} baris)")

if __name__ == '__main__':
    preprocess_dataset()
