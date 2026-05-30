import json
import numpy as np

with open(r"C:\Users\legol\OneDrive\바탕 화면\project_v5\json_dump.txt", "r", encoding="utf-8") as f:
    data = json.load(f)

face = data["annotations"]["joint_face"]
face_pts = np.array(face).reshape(-1, 2)

pose = data["annotations"]["joint_pose"]
pose_pts = np.array(pose).reshape(-1, 2)

# Find Chin (max Y in face)
chin_idx = np.argmax(face_pts[:, 1])

# Find Nose (closest to mean of face)
mean_face = np.mean(face_pts, axis=0)
nose_idx = np.argmin(np.linalg.norm(face_pts - mean_face, axis=1))

# Find Right Eye (min X among upper half of face)
# "Right eye" of the person is on the left side of the image (min X)
upper_half = face_pts[:, 1] < mean_face[1]
right_eye_candidates = np.where(upper_half)[0]
# Among upper half, find the one with minimum X
right_eye_idx = right_eye_candidates[np.argmin(face_pts[right_eye_candidates, 0])]

# Find Left Eye (max X among upper half of face)
left_eye_candidates = np.where(upper_half)[0]
left_eye_idx = left_eye_candidates[np.argmax(face_pts[left_eye_candidates, 0])]

print(f"Heuristic Face Indices:")
print(f"Chin: {chin_idx}")
print(f"Nose: {nose_idx}")
print(f"Right Eye: {right_eye_idx}")
print(f"Left Eye: {left_eye_idx}")

# For shoulders, usually they are the outermost points in the upper body.
# Let's just print the pose points to see.
print("\nPose points:")
for i, pt in enumerate(pose_pts):
    print(f"{i}: {pt}")
