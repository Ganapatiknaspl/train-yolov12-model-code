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
import atexit
import signal
from datetime import datetime
from ultralytics import YOLO
from github_sync import GitHubSync

# ==========================================================
# Configuration
# ==========================================================

DATASET_REPO = "https://github.com/Ganapatiknaspl/train-yolo12l-model.git"
CODE_REPO = "https://github.com/Ganapatiknaspl/train-yolov12-model-code.git"

DATASET_DIR = "AerialDataset"
CODE_DIR = "YOLO_Config"

MODEL_NAME = "yolo12l.pt"

PROJECT_NAME = "YOLO12_Training"
RUN_NAME = "Aircraft_v12l"

# ==========================================================
# GitHub Sync Configuration
# ==========================================================

# Destination GitHub repo where ALL training outputs are pushed:
#   - weights/best.pt, weights/last.pt
#   - weights/epoch10.pt, epoch20.pt, etc. (all periodic checkpoints)
#   - args.yaml, results.csv, confusion_matrix.png, and all plots
#
# AUTHENTICATION — set this environment variable before running:
#   On lightning.ai:  export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   Or set it in the studio secrets / environment panel.
SYNC_REPO = "https://github.com/Ganapatiknaspl/train-yolo12l-model"

# How often (in epochs) to push a mid-training checkpoint to GitHub.
# Should match or be a multiple of save_period= in model.train().
SYNC_EVERY_N_EPOCHS = 10

# ==========================================================
# Setup Logging
# ==========================================================

os.makedirs("logs", exist_ok=True)

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "logs.txt"), encoding='utf-8'),
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
    error_msg = "Git is not installed. Please install Git from: https://git-scm.com/downloads"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

logger.info("Git Found: %s", shutil.which("git"))

# ==========================================================
# Step 1: Clone/Pull Code Repository (Always get latest code)
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 1: Handling Code Repository")
logger.info("=" * 70)

if not os.path.exists(CODE_DIR):
    logger.info("Code repository not found. Cloning...")
    subprocess.run(
        ["git", "clone", "--depth", "1", CODE_REPO, CODE_DIR],
        check=True
    )
    logger.info("Code repository cloned successfully to: %s", CODE_DIR)
else:
    logger.info("Code repository exists. Pulling latest changes...")
    subprocess.run(
        ["git", "-C", CODE_DIR, "pull"],
        check=True
    )
    logger.info("Code repository updated successfully")

# ==========================================================
# Step 2: Read config.yaml from Code Repository
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 2: Reading config.yaml")
logger.info("=" * 70)

config_path = os.path.join(CODE_DIR, "config.yaml")

if not os.path.exists(config_path):
    logger.warning("config.yaml not found in code repository. Using default status=False")
    config = {"status": False}
else:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("config.yaml loaded: %s", config)

status = config.get("status", False)
logger.info("Status value: %s", status)

# ==========================================================
# Step 3: Handle Dataset Repository based on status
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 3: Handling Dataset Repository")
logger.info("=" * 70)

if not os.path.exists(DATASET_DIR):
    # Dataset directory doesn't exist locally
    if status:
        # Status is True - Clone the dataset
        logger.info("Dataset not found locally. Status is True. Cloning dataset...")
        subprocess.run(
            ["git", "clone", "--depth", "1", DATASET_REPO, DATASET_DIR],
            check=True
        )
        logger.info("Dataset cloned successfully to: %s", DATASET_DIR)
    else:
        # Status is False - Don't clone, raise error
        error_msg = (
            f"Dataset directory '{DATASET_DIR}' not found locally and status is False. "
            "Cannot proceed without dataset. Set status to True in config.yaml to clone it."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
else:
    # Dataset directory exists locally
    if status:
        # Status is True - Pull latest changes
        logger.info("Dataset found locally. Status is True. Pulling latest changes...")
        subprocess.run(
            ["git", "-C", DATASET_DIR, "pull"],
            check=True
        )
        logger.info("Dataset updated successfully")
    else:
        # Status is False - Use existing dataset
        logger.info("Dataset found locally. Status is False. Using existing dataset without updates")

# ==========================================================
# Step 4: Locate and Update data.yaml
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 4: Configuring data.yaml")
logger.info("=" * 70)

data_yaml_path = os.path.join(CODE_DIR, "data.yaml")

if not os.path.exists(data_yaml_path):
    error_msg = f"data.yaml not found at: {data_yaml_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

logger.info("data.yaml found at: %s", data_yaml_path)

# Read data.yaml
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Update path to absolute path of dataset
abs_dataset_path = os.path.abspath(DATASET_DIR)
data_config['path'] = abs_dataset_path

# Save updated data.yaml
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, sort_keys=False)

logger.info("Updated data.yaml path to: %s", abs_dataset_path)
logger.info("Dataset configuration:")
logger.info("  Train: %s", data_config.get('train', 'N/A'))
logger.info("  Valid: %s", data_config.get('val', 'N/A'))
logger.info("  Test: %s", data_config.get('test', 'N/A'))
logger.info("  Classes: %s", data_config.get('nc', 'N/A'))

# ==========================================================
# Step 5: Dataset Integrity Check
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 5: Checking Dataset Integrity")
logger.info("=" * 70)

total_removed = 0

for split in ['train', 'valid', 'test']:
    labels_dir = os.path.join(DATASET_DIR, split, 'labels')
    images_dir = os.path.join(DATASET_DIR, split, 'images')
    
    if os.path.exists(labels_dir) and os.path.exists(images_dir):
        removed = 0
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                # Check for corresponding image (.jpg or .png)
                image_name_jpg = label_file.replace('.txt', '.jpg')
                image_name_png = label_file.replace('.txt', '.png')
                
                image_exists = (
                    os.path.exists(os.path.join(images_dir, image_name_jpg)) or
                    os.path.exists(os.path.join(images_dir, image_name_png))
                )
                
                if not image_exists:
                    os.remove(os.path.join(labels_dir, label_file))
                    removed += 1
        
        if removed > 0:
            logger.info("  Removed %d orphaned label files from %s", removed, split)
            total_removed += removed

if total_removed > 0:
    logger.info("Total orphaned labels removed: %d", total_removed)
else:
    logger.info("Dataset integrity check passed - no orphaned labels found")

# ==========================================================
# Step 6: Initialize Metrics JSON
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 6: Setting up Metrics Tracking")
logger.info("=" * 70)

metrics_file = os.path.join("logs", "training_metrics.json")
metrics_data = {
    "training_info": {
        "start_time": None,
        "end_time": None,
        "total_epochs": 0,
        "model": MODEL_NAME,
        "dataset_repo": DATASET_REPO,
        "code_repo": CODE_REPO,
        "project": PROJECT_NAME,
        "run_name": RUN_NAME,
        "dataset_path": abs_dataset_path
    },
    "epochs": [],
    "final_metrics": {}
}

logger.info("Metrics will be saved to: %s", metrics_file)

# ==========================================================
# Step 7: Start Training
# ==========================================================

start_datetime = datetime.now()
start_time = time.time()

metrics_data["training_info"]["start_time"] = start_datetime.isoformat()

# Save initial metrics file
with open(metrics_file, 'w') as f:
    json.dump(metrics_data, f, indent=2, default=str)

logger.info("\n" + "=" * 70)
logger.info("Step 7: Starting Training")
logger.info("=" * 70)
logger.info("Training Start Time: %s", start_datetime)
logger.info("Model: %s", MODEL_NAME)
logger.info("Data YAML: %s", data_yaml_path)
logger.info("Project: %s", PROJECT_NAME)
logger.info("Run Name: %s", RUN_NAME)
logger.info("=" * 70)

# Load YOLO model
logger.info("Loading YOLO model: %s", MODEL_NAME)
model = YOLO(MODEL_NAME)
logger.info("Model loaded successfully")

# ==========================================================
# Step 7b: Setup GitHub Sync
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 7b: Setting up GitHub Sync")
logger.info("=" * 70)

# YOLO saves its run outputs to:  {PROJECT_NAME}/{RUN_NAME}/
yolo_run_dir = os.path.join(PROJECT_NAME, RUN_NAME)

# Initialize the sync engine (see github_sync.py)
github_syncer = None
try:
    github_syncer = GitHubSync(
        repo_url=SYNC_REPO,
        local_run_dir=yolo_run_dir,
        project=PROJECT_NAME,
        run_name=RUN_NAME,
    )
    github_syncer.setup()   # clone / pull the destination repo
    logger.info("GitHub sync ready. Outputs will be pushed to: %s", SYNC_REPO)
except Exception as _sync_setup_err:
    logger.error(
        "GitHub sync setup failed: %s\n"
        "Training will continue but outputs will NOT be synced to GitHub.",
        _sync_setup_err,
    )
    github_syncer = None

# ── Emergency push handlers ─────────────────────────────────────────────────
# Registered BEFORE training starts so that even if lightning.ai kills the
# process mid-epoch, the last saved checkpoint is pushed to GitHub.

def _emergency_push():
    """Called on any exit: normal, sys.exit(), Ctrl+C, or SIGTERM."""
    if github_syncer is not None:
        logger.info("Exit handler triggered — pushing training outputs to GitHub...")
        github_syncer.push_emergency()

# Python atexit: fires on normal exit AND uncaught exceptions
atexit.register(_emergency_push)

# SIGTERM: lightning.ai / Kubernetes / SLURM send this before killing a job
def _sigterm_handler(signum, frame):
    logger.info("SIGTERM received — triggering emergency GitHub push...")
    _emergency_push()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)
logger.info("Emergency push handlers registered (atexit + SIGTERM).")

# Define callback for epoch metrics
def log_epoch_metrics(trainer):
    """Callback function to save metrics after each epoch"""
    epoch = trainer.epoch
    metrics = trainer.metrics
    
    # Clean metrics for JSON serialization
    clean_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            clean_metrics[key] = value.item()
        else:
            clean_metrics[key] = value
    
    # Create epoch data
    epoch_data = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "metrics": clean_metrics
    }
    
    # Add to metrics data
    metrics_data["epochs"].append(epoch_data)
    metrics_data["training_info"]["total_epochs"] = epoch
    
    # Save metrics after each epoch
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)
    
    # Log epoch completion
    logger.info("Epoch %d completed - Metrics saved", epoch)

# Add callback to model
model.add_callback("on_fit_epoch_end", log_epoch_metrics)

# ── Periodic GitHub sync callback ────────────────────────────────────────────
# Fires every SYNC_EVERY_N_EPOCHS epochs.
# Pushes whatever files YOLO has saved so far (epoch*.pt, best.pt, last.pt,
# plots, metrics) directly to GitHub — so progress is safe even mid-training.

def sync_to_github(trainer):
    """YOLO callback: push training outputs to GitHub every N epochs."""
    epoch = trainer.epoch
    if github_syncer is not None and (epoch + 1) % SYNC_EVERY_N_EPOCHS == 0:
        logger.info("Epoch %d: Triggering periodic GitHub sync...", epoch)
        try:
            github_syncer.push_checkpoint(epoch)
            logger.info("Epoch %d: GitHub sync completed successfully.", epoch)
        except Exception as _sync_err:
            # Never let a sync error crash the training run
            logger.error(
                "Epoch %d: GitHub sync failed (training continues): %s",
                epoch, _sync_err,
            )

model.add_callback("on_fit_epoch_end", sync_to_github)
logger.info(
    "Periodic GitHub sync callback registered (every %d epochs).",
    SYNC_EVERY_N_EPOCHS,
)

# Train the model
logger.info("Starting training process...")

try:
    results = model.train(
        data=data_yaml_path,
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
    
    logger.info("Training completed successfully!")

    # ── Final GitHub push ────────────────────────────────────────────────────
    # Push the complete run folder to GitHub one last time after training ends.
    # This captures the final best.pt and any files not yet pushed mid-training.
    if github_syncer is not None:
        logger.info("Training complete — performing final GitHub push...")
        try:
            github_syncer.push_final()
            logger.info(
                "Final GitHub push completed! View at: %s",
                SYNC_REPO + f"/tree/main/runs/{PROJECT_NAME}/{RUN_NAME}",
            )
        except Exception as _final_push_err:
            logger.error("Final GitHub push failed: %s", _final_push_err)
    
except Exception as e:
    logger.error("Training failed: %s", str(e))
    raise

# ==========================================================
# Step 8: Save Final Metrics
# ==========================================================

logger.info("\n" + "=" * 70)
logger.info("Step 8: Saving Final Metrics")
logger.info("=" * 70)

# Extract final metrics
final_metrics = {}
try:
    if hasattr(results, 'results_dict'):
        final_metrics = results.results_dict
    elif hasattr(results, 'metrics'):
        final_metrics = results.metrics
except Exception as e:
    logger.warning("Could not extract final metrics: %s", e)

metrics_data["final_metrics"] = final_metrics
metrics_data["training_info"]["end_time"] = datetime.now().isoformat()

# Save final metrics
with open(metrics_file, 'w') as f:
    json.dump(metrics_data, f, indent=2, default=str)

logger.info("Final metrics saved to: %s", metrics_file)

# ==========================================================
# Step 9: Training Summary
# ==========================================================

end_datetime = datetime.now()
elapsed = time.time() - start_time

hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

logger.info("\n" + "=" * 70)
logger.info("Training Summary")
logger.info("=" * 70)
logger.info("Start Time: %s", start_datetime)
logger.info("End Time: %s", end_datetime)
logger.info("Duration: %d Hours %d Minutes %d Seconds", hours, minutes, seconds)
logger.info("Total Seconds: %.2f", elapsed)
logger.info("Total Epochs: %d", metrics_data['training_info']['total_epochs'])
logger.info("Model: %s", MODEL_NAME)
logger.info("Logs saved to: logs/logs.txt")
logger.info("Metrics saved to: %s", metrics_file)
logger.info("Trained model saved to: %s/%s/", PROJECT_NAME, RUN_NAME)
logger.info("=" * 70)

logger.info("Script completed successfully!")