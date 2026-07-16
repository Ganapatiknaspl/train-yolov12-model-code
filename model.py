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
from datetime import datetime

import yaml
from ultralytics import YOLO

# ==========================================================
# CONFIG FILE
# ==========================================================

CONFIG_FILE = "config.yaml"

# ==========================================================
# LOAD CONFIG
# ==========================================================

def load_config():

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(CONFIG_FILE)

    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

# ==========================================================
# SAVE CONFIG
# ==========================================================

def save_config(config):

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(
            config,
            f,
            sort_keys=False
        )

# ==========================================================
# VERIFY GIT
# ==========================================================

def verify_git():

    if shutil.which("git") is None:
        raise RuntimeError(
            "Git is not installed.\n"
            "https://git-scm.com/downloads"
        )

# ==========================================================
# CLONE DATASET
# ==========================================================

def clone_dataset(config):

    repo = config["dataset"]["repo"]
    folder = config["dataset"]["folder"]
    branch = config["dataset"]["branch"]

    already_cloned = config["status"]["cloned"]

    if already_cloned and os.path.isdir(folder):

        print("=" * 70)
        print("Repository already cloned.")
        print("=" * 70)
        return

    if os.path.isdir(folder):
        shutil.rmtree(folder)

    print("=" * 70)
    print("Cloning Dataset Repository...")
    print("=" * 70)

    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--depth",
            "1",
            repo,
            folder,
        ],
        check=True,
    )

    config["status"]["cloned"] = True

    save_config(config)

    print("=" * 70)
    print("Repository Cloned Successfully.")
    print("=" * 70)

# ==========================================================
# VERIFY DATASET
# ==========================================================

def verify_dataset(config):

    root = config["dataset"]["folder"]

    folders = [

        "train/images",
        "train/labels",

        "valid/images",
        "valid/labels",

        "test/images",
        "test/labels",
    ]

    for folder in folders:

        path = os.path.join(root, folder)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

    yaml_file = os.path.join(root, "data.yaml")

    if not os.path.exists(yaml_file):
        raise FileNotFoundError(yaml_file)

    return yaml_file

# ==========================================================
# TRAIN MODEL
# ==========================================================

def train(data_yaml):

    model = YOLO("yolo12l.pt")

    model.train(

        data=data_yaml,

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

        project="YOLO12_Training",

        name="Aircraft_v12l",

        exist_ok=True,
    )

# ==========================================================
# MAIN
# ==========================================================

def main():

    verify_git()

    config = load_config()

    clone_dataset(config)

    data_yaml = verify_dataset(config)

    start_datetime = datetime.now()

    start_time = time.time()

    print("=" * 70)
    print("Training Started")
    print(start_datetime)
    print("=" * 70)

    train(data_yaml)

    end_datetime = datetime.now()

    elapsed = time.time() - start_time

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print("=" * 70)
    print("Training Finished")
    print("=" * 70)

    print("Started :", start_datetime)
    print("Ended   :", end_datetime)

    print(f"Duration : {h}h {m}m {s}s")

    print("=" * 70)

# ==========================================================

if __name__ == "__main__":
    main()