import pandas as pd
import os
import zipfile
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'results.csv')
ZIP_FILE = os.path.join(PROJECT_ROOT, 'data', 'archive.zip')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_images')
TARGET_LIST = os.path.join(PROJECT_ROOT, 'data', 'real_images_list.csv')

ROLE_PATTERNS = {
    'doctor': r'\b(?:doctor|physician|surgeon|medical staff)\b',
    'nurse': r'\b(?:nurse|nursing)\b',
    'teacher': r'\b(?:teacher|professor|classroom|teaching)\b',
    'scientist': r'\b(?:scientist|researcher|lab coat|laboratory)\b',
    'chef': r'\b(?:chef|cook|cooking|kitchen staff)\b',
    'artist': r'\b(?:artist|painter|sculptor|painting|drawing)\b',
    'builder': r'\b(?:builder|construction|hard hat|safety vest)\b',
    'driver': r'\b(?:driver|taxi|bus driver|truck driver)\b'
}


def step_one_filter():
    print("1. Filtering Captions...")
    try:
        df = pd.read_csv(INPUT_CSV, delimiter='|', skiprows=1, names=['image_id', 'x', 'caption'])
    except:
        df = pd.read_csv(INPUT_CSV, delimiter='|', names=['image_id', 'x', 'caption'])

    df['caption'] = df['caption'].fillna('').astype(str).str.lower()
    all_hits = []

    for role, pattern in ROLE_PATTERNS.items():
        mask = df['caption'].str.contains(pattern, regex=True)
        hits = df.loc[mask, ['image_id']].copy()
        hits['role'] = role
        all_hits.append(hits)

    final_df = pd.concat(all_hits).drop_duplicates(subset=['image_id'])
    final_df.to_csv(TARGET_LIST, index=False)
    print(f"   -> Found {len(final_df)} images.")
    return final_df


def step_two_extract(df):
    print("2. Extracting Images from Zip...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    required = set(df['image_id'].astype(str))

    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        zip_map = {os.path.basename(f): f for f in z.namelist() if f.lower().endswith('.jpg')}

        for img in tqdm(required):
            if img in zip_map and not os.path.exists(os.path.join(OUTPUT_DIR, img)):
                with open(os.path.join(OUTPUT_DIR, img), 'wb') as f:
                    f.write(z.read(zip_map[img]))


if __name__ == "__main__":
    df = step_one_filter()
    step_two_extract(df)