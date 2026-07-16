import os
import subprocess
import shutil
import yaml
import time
from datetime import datetime
from ultralytics import YOLO

# ==========================================================
# Configuration
# ==========================================================

DATASET_REPO = "https://github.com/Ganapatiknaspl/train-yolo12l-model.git"
YAML_REPO = "https://github.com/Ganapatiknaspl/train-yolov12-model-code.git"

DATASET_DIR = "AerialDataset"
YAML_DIR = "YOLO_Config"

MODEL_NAME = "yolo12l.pt"

# ==========================================================
# Verify Git
# ==========================================================

if shutil.which("git") is None:
    raise RuntimeError("Git is not installed.")

# ==========================================================
# Clone Dataset Repository
# ==========================================================

if not os.path.exists(DATASET_DIR):

    print("Cloning Dataset Repository...")

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            DATASET_REPO,
            DATASET_DIR,
        ],
        check=True,
    )

else:

    print("Updating Dataset Repository...")

    subprocess.run(
        [
            "git",
            "-C",
            DATASET_DIR,
            "pull",
        ],
        check=True,
    )

# ==========================================================
# Clone YAML Repository
# ==========================================================

if not os.path.exists(YAML_DIR):

    print("Cloning Configuration Repository...")

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            YAML_REPO,
            YAML_DIR,
        ],
        check=True,
    )

else:

    print("Updating Configuration Repository...")

    subprocess.run(
        [
            "git",
            "-C",
            YAML_DIR,
            "pull",
        ],
        check=True,
    )

# ==========================================================
# Locate data.yaml
# ==========================================================

DATA_YAML = os.path.join(YAML_DIR, "data.yaml")

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(DATA_YAML)

# ==========================================================
# Update path inside YAML
# ==========================================================

with open(DATA_YAML, "r") as f:
    config = yaml.safe_load(f)

config["path"] = os.path.abspath(DATASET_DIR)

with open(DATA_YAML, "w") as f:
    yaml.dump(config, f, sort_keys=False)

print("Dataset Path Updated:")
print(config["path"])

# ==========================================================
# Start Timer
# ==========================================================

start_datetime = datetime.now()
start_time = time.time()

print("=" * 70)
print(f"Training Started : {start_datetime}")
print("=" * 70)

# ==========================================================
# Load Model
# ==========================================================

model = YOLO(MODEL_NAME)

# ==========================================================
# Train
# ==========================================================

results = model.train(

    data=DATA_YAML,

    epochs=100,
    patience=20,

    imgsz=1024,
    batch=16,
    workers=8,

    optimizer="AdamW",

    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.001,

    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=10,
    translate=0.10,
    scale=0.50,
    shear=2.0,
    perspective=0.0005,

    flipud=0.20,
    fliplr=0.50,

    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.30,
    close_mosaic=15,

    box=7.5,
    cls=1.5,
    dfl=1.5,

    dropout=0.10,
    label_smoothing=0.05,

    amp=True,
    cache=True,

    deterministic=True,
    seed=42,

    save=True,
    save_period=10,
    plots=True,

    project="./YOLO12_Training",
    name="Aircraft_v12l",
    exist_ok=True,
)

# ==========================================================
# End Timer
# ==========================================================

end_datetime = datetime.now()
elapsed = time.time() - start_time

hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print("=" * 70)
print(f"Training Started : {start_datetime}")
print(f"Training Ended   : {end_datetime}")
print(f"Duration         : {hours} Hours {minutes} Minutes {seconds} Seconds")
print("=" * 70)