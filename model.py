import os
import urllib.request
from urllib.parse import urlparse
from ultralytics import YOLO
from datetime import datetime
import time

# ==========================================================
# Repository Configuration
# ==========================================================

GITHUB_REPOSITORY = "https://github.com/Ganapatiknaspl/train-yolo12l-model"

# DATA_YAML_URL = (
#     GITHUB_REPOSITORY +
#     "./data.yaml"
# )

LOCAL_DATA_YAML = "data.yaml"

# ==========================================================
# Validate GitHub URL
# ==========================================================

# parsed = urlparse(GITHUB_REPOSITORY)

# if parsed.scheme not in ("http", "https"):
#     raise ValueError("Invalid GitHub URL")

# if "github.com" not in parsed.netloc.lower():
#     raise ValueError("Repository must be hosted on GitHub")

print("=" * 70)
print("GitHub Repository Verified")
print(GITHUB_REPOSITORY)
print("=" * 70)

# ==========================================================
# Download data.yaml if needed
# ==========================================================

if not os.path.exists(LOCAL_DATA_YAML):

    print("Downloading data.yaml from GitHub...")

    urllib.request.urlretrieve(
        LOCAL_DATA_YAML
    )

    print("Download completed.")

else:
    print("Using existing local data.yaml")

# ==========================================================
# Verify
# ==========================================================

if not os.path.exists(LOCAL_DATA_YAML):
    raise FileNotFoundError("data.yaml could not be downloaded.")

# ==========================================================
# Record Start Time
# ==========================================================

start_datetime = datetime.now()
start_time = time.time()

print("=" * 70)
print(f"Training Started : {start_datetime}")
print("=" * 70)

model = YOLO("yolo12l.pt")

results = model.train(
    data=LOCAL_DATA_YAML,

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
# Training Summary
# ==========================================================

end_datetime = datetime.now()
elapsed = time.time() - start_time

print("=" * 70)
print(f"Started : {start_datetime}")
print(f"Ended   : {end_datetime}")
print(f"Duration: {elapsed/3600:.2f} Hours")
print("=" * 70)