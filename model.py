# import os
# import shutil
# import subprocess
# import time
# from datetime import datetime

# from ultralytics import YOLO

# # ==========================================================
# # Configuration
# # ==========================================================

# # GitHub repository containing:
# # data.yaml
# # train/
# # valid/
# # test/

# DATASET_REPO = "https://github.com/Ganapatiknaspl/train-yolo12l-model.git"

# LOCAL_DATASET_DIR = "AerialDataset"

# MODEL = "yolo12l.pt"

# PROJECT_NAME = "YOLO12_Training"
# RUN_NAME = "Aircraft_v12l"

# # ==========================================================
# # Verify Git Installation
# # ==========================================================

# print("=" * 70)
# print("Checking Git Installation...")
# print("=" * 70)

# if shutil.which("git") is None:
#     raise RuntimeError(
#         "\nGit is not installed.\n"
#         "Download from: https://git-scm.com/downloads"
#     )

# print("Git Found")

# # ==========================================================
# # Clone Repository
# # ==========================================================

# if not os.path.exists(LOCAL_DATASET_DIR):

#     print("\nDownloading Dataset Repository...\n")

#     subprocess.run(
#         [
#             "git",
#             "clone",
#             "--depth",
#             "1",
#             DATASET_REPO,
#             LOCAL_DATASET_DIR
#         ],
#         check=True
#     )

# else:

#     print("\nDataset already exists.")
#     print("Updating Repository...\n")

#     subprocess.run(
#         [
#             "git",
#             "-C",
#             LOCAL_DATASET_DIR,
#             "pull"
#         ],
#         check=True
#     )

# print("\nRepository Ready.")

# # ==========================================================
# # Locate data.yaml
# # ==========================================================

# DATA_YAML = os.path.join(
#     LOCAL_DATASET_DIR,
#     "data.yaml"
# )

# if not os.path.exists(DATA_YAML):
#     raise FileNotFoundError(
#         f"\nCannot locate:\n{DATA_YAML}"
#     )

# print(f"\ndata.yaml Found\n{DATA_YAML}")

# # ==========================================================
# # Start Timer
# # ==========================================================

# start_datetime = datetime.now()
# start_time = time.time()

# print("\n" + "=" * 70)
# print(f"Training Started : {start_datetime}")
# print("=" * 70)

# # ==========================================================
# # Load YOLO Model
# # ==========================================================

# model = YOLO(MODEL)

# # ==========================================================
# # Train Model
# # ==========================================================

# results = model.train(

#     # Dataset
#     data=DATA_YAML,

#     # ------------------------------------------------------
#     # Training
#     # ------------------------------------------------------

#     epochs=100,
#     patience=20,

#     imgsz=1024,

#     batch=16,

#     workers=8,

#     # ------------------------------------------------------
#     # Optimizer
#     # ------------------------------------------------------

#     optimizer="AdamW",

#     lr0=0.001,

#     lrf=0.01,

#     momentum=0.937,

#     weight_decay=0.001,

#     # ------------------------------------------------------
#     # Warmup
#     # ------------------------------------------------------

#     warmup_epochs=3,

#     warmup_momentum=0.8,

#     warmup_bias_lr=0.1,

#     # ------------------------------------------------------
#     # Augmentation
#     # ------------------------------------------------------

#     hsv_h=0.015,

#     hsv_s=0.7,

#     hsv_v=0.4,

#     degrees=10,

#     translate=0.10,

#     scale=0.50,

#     shear=2.0,

#     perspective=0.0005,

#     flipud=0.20,

#     fliplr=0.50,

#     mosaic=1.0,

#     mixup=0.15,

#     copy_paste=0.30,

#     close_mosaic=15,

#     # ------------------------------------------------------
#     # Loss
#     # ------------------------------------------------------

#     box=7.5,

#     cls=1.5,

#     dfl=1.5,

#     # ------------------------------------------------------
#     # Regularization
#     # ------------------------------------------------------

#     dropout=0.10,

#     label_smoothing=0.05,

#     # ------------------------------------------------------
#     # Performance
#     # ------------------------------------------------------

#     amp=True,

#     cache=True,

#     deterministic=True,

#     seed=42,

#     # ------------------------------------------------------
#     # Save
#     # ------------------------------------------------------

#     save=True,

#     save_period=10,

#     plots=True,

#     project=PROJECT_NAME,

#     name=RUN_NAME,

#     exist_ok=True,
# )

# # ==========================================================
# # End Timer
# # ==========================================================

# end_datetime = datetime.now()

# elapsed = time.time() - start_time

# hours = int(elapsed // 3600)
# minutes = int((elapsed % 3600) // 60)
# seconds = int(elapsed % 60)

# print("\n" + "=" * 70)
# print("Training Completed")
# print("=" * 70)

# print(f"Started : {start_datetime}")
# print(f"Ended   : {end_datetime}")

# print(
#     f"Duration: {hours} Hours "
#     f"{minutes} Minutes "
#     f"{seconds} Seconds"
# )

# print(f"Total Seconds : {elapsed:.2f}")

# print("=" * 70)

import os
import shutil
import subprocess
import time
import yaml
import json
import logging
from datetime import datetime

from ultralytics import YOLO

# ==========================================================
# Configuration
# ==========================================================

DATASET_REPO = "https://github.com/Ganapatiknaspl/train-yolo12l-model.git"
LOCAL_DATASET_DIR = "AerialDataset"
MODEL = "yolo12l.pt"
PROJECT_NAME = "YOLO12_Training"
RUN_NAME = "Aircraft_v12l"

# ==========================================================
# Setup Logging
# ==========================================================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "training.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================================
# Verify Git Installation
# ==========================================================

logger.info("=" * 70)
logger.info("Checking Git Installation...")
logger.info("=" * 70)

if shutil.which("git") is None:
    raise RuntimeError(
        "\nGit is not installed.\n"
        "Download from: https://git-scm.com/downloads"
    )

logger.info("Git Found")

# ==========================================================
# Read status from config.yaml in the dataset repo
# ==========================================================

config_path = os.path.join(LOCAL_DATASET_DIR, "config.yaml")

# Check if dataset exists and has config.yaml
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        status = config.get("status", False)
    logger.info(f"Status from config.yaml: {status} (True = clone, False = skip)")
else:
    # If dataset doesn't exist, we need to clone it
    if not os.path.exists(LOCAL_DATASET_DIR):
        status = True
        logger.info("Dataset not found. Will clone.")
    else:
        status = False
        logger.warning("config.yaml not found. Defaulting to skip clone.")

# ==========================================================
# Clone Repository based on status
# ==========================================================

if status:  # True = CLONE
    logger.info("\nCloning repository...")
    if os.path.exists(LOCAL_DATASET_DIR):
        logger.info("Dataset already exists. Pulling latest changes...")
        subprocess.run(
            ["git", "-C", LOCAL_DATASET_DIR, "pull"],
            check=True
        )
    else:
        logger.info("Downloading Dataset Repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", DATASET_REPO, LOCAL_DATASET_DIR],
            check=True
        )
    logger.info("Repository ready.")
else:  # False = DON'T CLONE
    logger.info("Status is False. Skipping clone.")
    if not os.path.exists(LOCAL_DATASET_DIR):
        raise RuntimeError(
            f"Dataset not found at {LOCAL_DATASET_DIR} and status is False.\n"
            f"Set status: true in config.yaml to clone the dataset."
        )

# ==========================================================
# Locate data.yaml and update path
# ==========================================================

DATA_YAML = os.path.join(LOCAL_DATASET_DIR, "data.yaml")

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"\nCannot locate:\n{DATA_YAML}")

logger.info(f"\ndata.yaml Found\n{DATA_YAML}")

# Update data.yaml with absolute path
with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)

abs_path = os.path.abspath(LOCAL_DATASET_DIR)
data_config['path'] = abs_path

with open(DATA_YAML, 'w') as f:
    yaml.dump(data_config, f, sort_keys=False)

logger.info(f"Updated data.yaml path to: {abs_path}")

# ==========================================================
# Fix: Remove label files without corresponding images
# ==========================================================

logger.info("\nChecking dataset integrity...")

for split in ['train', 'valid', 'test']:
    labels_dir = os.path.join(LOCAL_DATASET_DIR, split, 'labels')
    images_dir = os.path.join(LOCAL_DATASET_DIR, split, 'images')
    
    if os.path.exists(labels_dir) and os.path.exists(images_dir):
        removed = 0
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                # Try .jpg
                image_name = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_name)
                
                # Try .png if .jpg not found
                if not os.path.exists(image_path):
                    image_name = label_file.replace('.txt', '.png')
                    image_path = os.path.join(images_dir, image_name)
                
                # If still not found, remove the label
                if not os.path.exists(image_path):
                    os.remove(os.path.join(labels_dir, label_file))
                    removed += 1
        
        if removed > 0:
            logger.info(f"  Removed {removed} label files without images in {split}")

logger.info("Dataset check complete.")

# ==========================================================
# Setup Metrics JSON
# ==========================================================

metrics_file = os.path.join("logs", "training_metrics.json")
metrics_data = {
    "training_info": {
        "start_time": None,
        "end_time": None,
        "total_epochs": 0,
        "model": MODEL,
        "dataset": DATA_YAML,
        "project": PROJECT_NAME,
        "run_name": RUN_NAME
    },
    "epochs": [],
    "final_metrics": {}
}

# ==========================================================
# Start Timer
# ==========================================================

start_datetime = datetime.now()
start_time = time.time()

metrics_data["training_info"]["start_time"] = start_datetime.isoformat()

logger.info("\n" + "=" * 70)
logger.info(f"Training Started : {start_datetime}")
logger.info("=" * 70)

# ==========================================================
# Load YOLO Model
# ==========================================================

model = YOLO(MODEL)

# ==========================================================
# Train Model with callback to log metrics
# ==========================================================

def log_epoch_metrics(trainer):
    epoch = trainer.epoch
    metrics = trainer.metrics
    
    # Clean metrics for JSON
    clean_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            clean_metrics[key] = value.item()
        else:
            clean_metrics[key] = value
    
    # Add to metrics data
    metrics_data["epochs"].append({
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "metrics": clean_metrics
    })
    metrics_data["training_info"]["total_epochs"] = epoch
    
    # Save to JSON after each epoch
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)
    
    logger.info(f"Epoch {epoch} metrics saved to JSON")

model.add_callback("on_fit_epoch_end", log_epoch_metrics)

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
    project=PROJECT_NAME,
    name=RUN_NAME,
    exist_ok=True,
)

# ==========================================================
# Save Final Metrics
# ==========================================================

final_metrics = {}
try:
    if hasattr(results, 'res_dict'):
        final_metrics = results.res_dict
    elif hasattr(results, 'metrics'):
        final_metrics = results.metrics
except:
    pass

metrics_data["final_metrics"] = final_metrics
metrics_data["training_info"]["end_time"] = datetime.now().isoformat()

with open(metrics_file, 'w') as f:
    json.dump(metrics_data, f, indent=2, default=str)

logger.info(f"\nMetrics saved to: {metrics_file}")

# ==========================================================
# End Timer
# ==========================================================

end_datetime = datetime.now()
elapsed = time.time() - start_time

hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

logger.info("\n" + "=" * 70)
logger.info("Training Completed")
logger.info("=" * 70)
logger.info(f"Started : {start_datetime}")
logger.info(f"Ended   : {end_datetime}")
logger.info(f"Duration: {hours} Hours {minutes} Minutes {seconds} Seconds")
logger.info(f"Total Seconds : {elapsed:.2f}")
logger.info("=" * 70)