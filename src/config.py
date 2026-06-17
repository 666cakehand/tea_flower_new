import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "tea_flower_datasets")
IMAGES_DIR = os.path.join(DATA_DIR, "茶花图片合集")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-cls.pt")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

FEATURES = ["花型", "花瓣皱褶", "花瓣外侧主色的颜色", "花瓣着色类型"]

SEED = 42

TRAIN_PARAMS = {
    "batch_size": 16,
    "epochs": 10,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 4,
    "img_size": 224,
    "patience": 5,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)