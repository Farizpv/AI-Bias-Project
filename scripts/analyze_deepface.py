import pandas as pd
import os
from deepface import DeepFace
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_images')
INPUT_LIST = os.path.join(PROJECT_ROOT, 'data', 'real_images_list.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'analysis_real.csv')


def analyze_real():
    print("Analyzing REAL images...")
    df = pd.read_csv(INPUT_LIST)

    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(existing['image_id'].astype(str))
        print(f"   -> Resuming... ({len(done_ids)} already done)")
    else:
        done_ids = set()
        with open(OUTPUT_CSV, 'w') as f:
            f.write("image_id,role,face_detected,gender,race\n")

    with open(OUTPUT_CSV, 'a') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if str(row['image_id']) in done_ids: continue

            try:
                obj = DeepFace.analyze(os.path.join(IMAGE_DIR, str(row['image_id'])),
                                       actions=['gender', 'race'],
                                       detector_backend='opencv',
                                       silent=True)
                gender = max(obj[0]['gender'], key=obj[0]['gender'].get)
                race = obj[0]['dominant_race']
                f.write(f"{row['image_id']},{row['role']},True,{gender},{race}\n")
            except:
                f.write(f"{row['image_id']},{row['role']},False,unknown,unknown\n")


if __name__ == "__main__":
    analyze_real()