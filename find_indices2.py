import json
import numpy as np

with open(r"C:\Users\legol\OneDrive\바탕 화면\project_v5\json_dump.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

face = data["annotations"]["joint_face"]
face_pts = np.array(face).reshape(-1, 2)

print(f"Index 0: {face_pts[0]}")
print(f"Index 53: {face_pts[53]}")
print(f"Index 67: {face_pts[67]}")
print(f"Index 68: {face_pts[68]}")

print("---")
# Left Eye (person's left, viewer's right, larger X, small Y)
left_eye = np.argmax(face_pts[:, 0]) 
# Let's find the eye center by looking at the highest points
sorted_y = np.argsort(face_pts[:, 1])
# top 20 points
top_20 = face_pts[sorted_y[:20]]
top_20_indices = sorted_y[:20]
for idx, pt in zip(top_20_indices, top_20):
    print(f"Top idx {idx}: {pt}")
