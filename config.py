import os
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    # 주피터 노트북(Colab 셀)에서 직접 실행할 경우 __file__이 없으므로 현재 작업 폴더를 사용합니다.
    PROJECT_ROOT = Path.cwd().resolve()

# Keep this ASCII-safe. For Korean Windows paths, pass --dataset-root,
# --image-root, and --label-root in the command line, or set DATASET_ROOT.
if Path("/content").exists():
    _default_dataset_root = Path("/content/030.한국인 전신 및 포즈 데이터")
else:
    _default_dataset_root = Path(r"C:\Users\legol\OneDrive\바탕 화면\project_v5\030.한국인 전신 및 포즈 데이터")

DATASET_ROOT = Path(os.environ.get("TURTLE_NECK_DATASET_ROOT", _default_dataset_root))

IMAGE_SIZE = 256
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Use "front" for the realistic laptop webcam demo. "side" keeps the older
# proxy-CVA interpretation for side-view experiments.
POSTURE_MODE = "front"

# IMPORTANT:
# These are starter proxy points from one inspected AIHub-style sample.
# You MUST run scripts/inspect_annotations.py and visualize_keypoints.py, then
# verify/edit these indices for your final dataset split.
TARGET_KEYPOINTS = [
    {
        "name": "left_eye",
        "source": "joint_face",
        "index": 67,
        "description": "Subject's left eye (image right, larger X).",
    },
    {
        "name": "right_eye",
        "source": "joint_face",
        "index": 68,
        "description": "Subject's right eye (image left, smaller X).",
    },
    {
        "name": "nose",
        "source": "joint_face",
        "index": 0,
        "description": "Nose center.",
    },
    {
        "name": "chin",
        "source": "joint_face",
        "index": 53,
        "description": "Chin/lower face proxy for face height.",
    },
    {
        "name": "left_shoulder",
        "source": "joint_pose",
        "index": 22,
        "description": "Subject's left shoulder.",
    },
    {
        "name": "right_shoulder",
        "source": "joint_pose",
        "index": 19,
        "description": "Subject's right shoulder.",
    },
]

CVA_EAR_POINT = "nose"
CVA_C7_MODE = "midpoint"
CVA_C7_POINTS = ("left_shoulder", "right_shoulder")

GOOD_CVA_DEG = 50.0
WARN_CVA_DEG = 45.0
BAD_ALERT_SECONDS = 3.0

# Front-view proxy posture thresholds. Score is 0-100; higher is worse.
FRONT_GOOD_MAX_SCORE = 35.0
FRONT_WARN_MAX_SCORE = 65.0
FRONT_PSEUDO_Z_BAD_RATIO = 0.18
FRONT_HEAD_DROP_BAD_RATIO = 0.18
FRONT_SHOULDER_TILT_BAD_DEG = 8.0
FRONT_RATIO_BAD_DROP = 0.18

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
BACKBONE = "resnet18"
USE_PRETRAINED = False

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"
