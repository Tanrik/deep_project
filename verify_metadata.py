import pandas as pd
import json
import cv2
import numpy as np
import random
from pathlib import Path
import os

# Create output dir in artifacts folder so the user can easily view them
artifact_dir = Path(r"C:\Users\legol\.gemini\antigravity\brain\9edf92cb-7ecd-4443-8d44-4dedfd8cc31d\verification")
artifact_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(r"C:\Users\legol\OneDrive\바탕 화면\project_v5\data\metadata.csv")

# Filter valid rows
mask = ~df["keypoints"].str.contains(r"\[0\.0,\s*0\.0\]", regex=True)
df = df[mask]

# Sample 5 random rows
sample_df = df.sample(n=5, random_state=42)

colors = [
    (0, 0, 255),   # 0: Left Eye (Red)
    (0, 255, 0),   # 1: Right Eye (Green)
    (255, 0, 0),   # 2: Nose (Blue)
    (0, 255, 255), # 3: Chin (Yellow)
    (255, 255, 0), # 4: Left Shoulder (Cyan)
    (255, 0, 255)  # 5: Right Shoulder (Magenta)
]

names = ["L-Eye", "R-Eye", "Nose", "Chin", "L-Sho", "R-Sho"]

for i, (_, row) in enumerate(sample_df.iterrows()):
    img_path = row["image_path"]
    kps = json.loads(row["keypoints"])
    
    # AIHub paths might need fixing depending on how they are saved
    # Let's check if the file exists
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
        
    try:
        n = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Failed to read file: {e}")
        continue
        
    if img is None:
        print(f"Failed to decode image: {img_path}")
        continue
        
    h, w = img.shape[:2]
    
    for idx, pt in enumerate(kps):
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(img, (x, y), 8, colors[idx], -1)
        cv2.putText(img, f"{idx}:{names[idx]}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[idx], 2)
        
    out_path = artifact_dir / f"verify_{i}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path}")

print("Verification images generated successfully!")
