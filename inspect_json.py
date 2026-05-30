import json
from pathlib import Path

root = Path(r"C:\Users\legol\OneDrive\바탕 화면\project_v5\030.한국인 전신 및 포즈 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터")
for file in root.rglob("*.json"):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
            if "joint_face" in data or "joint_pose" in data:
                with open(r"C:\Users\legol\OneDrive\바탕 화면\project_v5\json_dump.txt", "w", encoding="utf-8") as out:
                    out.write(data)
                print("Found:", file)
                break
    except Exception as e:
        pass
