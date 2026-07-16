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
"""
YOLO12L Training Pipeline

A complete production-level training pipeline that automatically:
- Clones the dataset repository if not present
- Updates the repository if already cloned
- Validates dataset structure
- Updates data.yaml with correct absolute paths
- Trains YOLO12L model with specified parameters
- Logs all operations and errors
"""

import os
import sys
import shutil
import subprocess
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from ultralytics import YOLO
from ultralytics.exceptions import UltralyticsError

# ==========================================================
# CONSTANTS
# ==========================================================

CONFIG_FILE = "config.yaml"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "training.log")

# ==========================================================
# DEFAULT CONFIGURATION
# ==========================================================

DEFAULT_CONFIG = {
    "dataset": {
        "repository": "https://github.com/Ganapatiknaspl/train-yolo12l-model.git",
        "local_folder": "AerialDataset",
        "branch": "master"
    },
    "training": {
        "model": "yolo12l.pt",
        "project": "YOLO12_Training",
        "run_name": "Aircraft_v12l"
    },
    "status": {
        "cloned": False,
        "last_updated": None
    }
}

# ==========================================================
# LOGGING SETUP
# ==========================================================

def setup_logging() -> logging.Logger:
    """Configure logging system with both file and console handlers."""
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("YOLOTraining")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ==========================================================
# CONFIGURATION MANAGER
# ==========================================================

class ConfigurationManager:
    """
    Manages configuration loading, saving, and validation.
    """
    
    def __init__(self, config_path: str = CONFIG_FILE, logger: Optional[logging.Logger] = None):
        """
        Initialize the ConfigurationManager.
        
        Args:
            config_path: Path to the configuration file
            logger: Logger instance for logging operations
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger("YOLOTraining")
        self.config: Dict[str, Any] = {}
        
    def create_default_config(self) -> None:
        """Create default configuration file if it doesn't exist."""
        if not os.path.exists(self.config_path):
            self.logger.info(f"Creating default configuration: {self.config_path}")
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(DEFAULT_CONFIG, f, sort_keys=False, default_flow_style=False)
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dict containing the configuration
            
        Raises:
            yaml.YAMLError: If the configuration file is invalid
        """
        self.create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            if self.config is None:
                self.config = DEFAULT_CONFIG.copy()
                self.save()
                
            self.logger.debug("Configuration loaded successfully")
            return self.config
            
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in config file: {e}")
            raise
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, sort_keys=False, default_flow_style=False)
            self.logger.debug("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def update_status(self, cloned: bool, last_updated: Optional[str] = None) -> None:
        """
        Update the status section of the configuration.
        
        Args:
            cloned: Whether the repository has been cloned
            last_updated: Timestamp of last update
        """
        if "status" not in self.config:
            self.config["status"] = {}
        
        self.config["status"]["cloned"] = cloned
        
        if last_updated is None:
            last_updated = datetime.now().isoformat()
        self.config["status"]["last_updated"] = last_updated
        
        self.save()
    
    def get_dataset_config(self) -> Dict[str, str]:
        """Get dataset configuration section."""
        return self.config.get("dataset", {})
    
    def get_training_config(self) -> Dict[str, str]:
        """Get training configuration section."""
        return self.config.get("training", {})

# ==========================================================
# GIT REPOSITORY MANAGER
# ==========================================================

class GitRepositoryManager:
    """
    Manages Git operations including clone, pull, and validation.
    """
    
    def __init__(
        self,
        repo_url: str,
        local_folder: str,
        branch: str = "master",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the GitRepositoryManager.
        
        Args:
            repo_url: Git repository URL
            local_folder: Local folder name for the repository
            branch: Branch to clone/pull
            logger: Logger instance for logging operations
        """
        self.repo_url = repo_url
        self.local_folder = local_folder
        self.branch = branch
        self.logger = logger or logging.getLogger("YOLOTraining")
    
    @staticmethod
    def verify_git_installed() -> None:
        """
        Verify that Git is installed on the system.
        
        Raises:
            RuntimeError: If Git is not installed
        """
        if shutil.which("git") is None:
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                "ERROR: Git is not installed on this system.\n"
                "Please install Git from: https://git-scm.com/downloads\n"
                "After installation, restart your terminal and try again.\n"
                "=" * 70
            )
    
    def is_repository_cloned(self) -> bool:
        """
        Check if the repository is already cloned.
        
        Returns:
            True if the repository exists locally
        """
        return os.path.exists(self.local_folder) and os.path.isdir(self.local_folder)
    
    def clone_repository(self) -> None:
        """
        Clone the repository from GitHub.
        
        Raises:
            subprocess.CalledProcessError: If git clone fails
            RuntimeError: If cloning fails
        """
        self.logger.info(f"Cloning repository from: {self.repo_url}")
        self.logger.info(f"Destination: {self.local_folder}")
        
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    self.branch,
                    self.repo_url,
                    self.local_folder
                ],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info("Repository cloned successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git clone failed: {e.stderr}")
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")
    
    def pull_repository(self) -> None:
        """
        Pull latest changes from the repository.
        
        Raises:
            subprocess.CalledProcessError: If git pull fails
            RuntimeError: If pulling fails
        """
        self.logger.info(f"Pulling latest changes from: {self.repo_url}")
        
        try:
            subprocess.run(
                ["git", "-C", self.local_folder, "pull", "origin", self.branch],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info("Repository updated successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git pull failed: {e.stderr}")
            raise RuntimeError(f"Failed to pull repository: {e.stderr}")
    
    def ensure_repository(self) -> None:
        """
        Ensure the repository is available locally.
        Clones if not present, pulls if already exists.
        """
        self.verify_git_installed()
        
        if self.is_repository_cloned():
            self.logger.info("Repository already exists locally")
            self.pull_repository()
        else:
            self.clone_repository()

# ==========================================================
# DATASET VALIDATOR
# ==========================================================

class DatasetValidator:
    """
    Validates the dataset structure and required files.
    """
    
    REQUIRED_PATHS = [
        "train/images",
        "train/labels",
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels"
    ]
    
    def __init__(self, dataset_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the DatasetValidator.
        
        Args:
            dataset_path: Path to the dataset directory
            logger: Logger instance for logging operations
        """
        self.dataset_path = Path(dataset_path)
        self.logger = logger or logging.getLogger("YOLOTraining")
    
    def validate_data_yaml(self) -> Path:
        """
        Locate and validate data.yaml file.
        
        Returns:
            Path to the data.yaml file
            
        Raises:
            FileNotFoundError: If data.yaml is not found
        """
        data_yaml_path = self.dataset_path / "data.yaml"
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"data.yaml not found in: {self.dataset_path}\n"
                f"Expected location: {data_yaml_path}"
            )
        
        self.logger.info(f"data.yaml found at: {data_yaml_path}")
        return data_yaml_path
    
    def validate_structure(self) -> None:
        """
        Validate the complete dataset structure.
        
        Raises:
            FileNotFoundError: If any required directory is missing
        """
        self.logger.info("Validating dataset structure...")
        
        missing_paths = []
        
        for path in self.REQUIRED_PATHS:
            full_path = self.dataset_path / path
            if not full_path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            error_msg = (
                f"Dataset validation failed. Missing required directories:\n"
                + "\n".join(f"  - {p}" for p in missing_paths)
                + f"\n\nExpected structure in: {self.dataset_path}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info("Dataset structure validated successfully")

# ==========================================================
# DATASET MANAGER
# ==========================================================

class DatasetManager:
    """
    Manages dataset operations including cloning, validation, and configuration.
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DatasetManager.
        
        Args:
            config_manager: ConfigurationManager instance
            logger: Logger instance for logging operations
        """
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger("YOLOTraining")
        self.git_manager: Optional[GitRepositoryManager] = None
        self.validator: Optional[DatasetValidator] = None
        
    def prepare_dataset(self) -> Path:
        """
        Prepare the dataset for training.
        
        Returns:
            Path to the dataset directory
            
        Raises:
            RuntimeError: If dataset preparation fails
        """
        dataset_config = self.config_manager.get_dataset_config()
        
        repo_url = dataset_config.get("repository")
        local_folder = dataset_config.get("local_folder")
        branch = dataset_config.get("branch", "master")
        
        if not repo_url or not local_folder:
            raise ValueError("Repository URL and local folder must be specified in config")
        
        # Initialize Git manager
        self.git_manager = GitRepositoryManager(
            repo_url=repo_url,
            local_folder=local_folder,
            branch=branch,
            logger=self.logger
        )
        
        # Ensure repository is available
        self.git_manager.ensure_repository()
        
        # Update status in config
        self.config_manager.update_status(cloned=True)
        
        # Get absolute path
        dataset_path = Path(os.path.abspath(local_folder))
        
        # Initialize validator
        self.validator = DatasetValidator(
            dataset_path=str(dataset_path),
            logger=self.logger
        )
        
        # Validate dataset
        data_yaml_path = self.validator.validate_data_yaml()
        self.validator.validate_structure()
        
        # Update data.yaml with absolute path
        self._update_data_yaml_path(data_yaml_path, str(dataset_path))
        
        return dataset_path
    
    def _update_data_yaml_path(self, data_yaml_path: Path, absolute_path: str) -> None:
        """
        Update the path in data.yaml to the absolute local directory.
        
        Args:
            data_yaml_path: Path to data.yaml
            absolute_path: Absolute path to the dataset directory
            
        Raises:
            RuntimeError: If updating data.yaml fails
        """
        try:
            self.logger.info(f"Updating data.yaml with absolute path: {absolute_path}")
            
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            
            if data_yaml is None:
                raise ValueError("data.yaml is empty or invalid")
            
            # Update the path
            data_yaml['path'] = absolute_path
            
            # Write back
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, sort_keys=False)
            
            self.logger.info("data.yaml updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update data.yaml: {e}")
            raise RuntimeError(f"Failed to update data.yaml: {e}")

# ==========================================================
# YOLO TRAINER
# ==========================================================

@dataclass
class TrainingParameters:
    """Training parameters for YOLO model."""
    
    epochs: int = 100
    patience: int = 20
    imgsz: int = 1024
    batch: int = 16
    workers: int = 8
    
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.001
    
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.10
    scale: float = 0.50
    shear: float = 2.0
    perspective: float = 0.0005
    flipud: float = 0.20
    fliplr: float = 0.50
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.30
    close_mosaic: int = 15
    
    box: float = 7.5
    cls: float = 1.5
    dfl: float = 1.5
    
    dropout: float = 0.10
    label_smoothing: float = 0.05
    
    amp: bool = True
    cache: bool = True
    deterministic: bool = True
    seed: int = 42
    
    save: bool = True
    save_period: int = 10
    plots: bool = True
    exist_ok: bool = True


class YOLOTrainer:
    """
    Manages YOLO model training.
    """
    
    def __init__(
        self,
        model_name: str,
        project_name: str,
        run_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the YOLOTrainer.
        
        Args:
            model_name: Name of the YOLO model (e.g., 'yolo12l.pt')
            project_name: Project name for training outputs
            run_name: Run name for this training session
            logger: Logger instance for logging operations
        """
        self.model_name = model_name
        self.project_name = project_name
        self.run_name = run_name
        self.logger = logger or logging.getLogger("YOLOTraining")
        self.model: Optional[YOLO] = None
        
    def load_model(self) -> None:
        """Load the YOLO model."""
        self.logger.info(f"Loading model: {self.model_name}")
        try:
            self.model = YOLO(self.model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def train(
        self,
        data_yaml: str,
        params: Optional[TrainingParameters] = None
    ) -> Any:
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to data.yaml file
            params: Training parameters
            
        Returns:
            Training results
            
        Raises:
            UltralyticsError: If training fails
            RuntimeError: If training fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if params is None:
            params = TrainingParameters()
        
        self.logger.info("Starting YOLO training...")
        self.logger.info(f"Data: {data_yaml}")
        self.logger.info(f"Epochs: {params.epochs}")
        self.logger.info(f"Batch size: {params.batch}")
        self.logger.info(f"Image size: {params.imgsz}")
        
        try:
            results = self.model.train(
                data=data_yaml,
                epochs=params.epochs,
                patience=params.patience,
                imgsz=params.imgsz,
                batch=params.batch,
                workers=params.workers,
                optimizer=params.optimizer,
                lr0=params.lr0,
                lrf=params.lrf,
                momentum=params.momentum,
                weight_decay=params.weight_decay,
                warmup_epochs=params.warmup_epochs,
                warmup_momentum=params.warmup_momentum,
                warmup_bias_lr=params.warmup_bias_lr,
                hsv_h=params.hsv_h,
                hsv_s=params.hsv_s,
                hsv_v=params.hsv_v,
                degrees=params.degrees,
                translate=params.translate,
                scale=params.scale,
                shear=params.shear,
                perspective=params.perspective,
                flipud=params.flipud,
                fliplr=params.fliplr,
                mosaic=params.mosaic,
                mixup=params.mixup,
                copy_paste=params.copy_paste,
                close_mosaic=params.close_mosaic,
                box=params.box,
                cls=params.cls,
                dfl=params.dfl,
                dropout=params.dropout,
                label_smoothing=params.label_smoothing,
                amp=params.amp,
                cache=params.cache,
                deterministic=params.deterministic,
                seed=params.seed,
                save=params.save,
                save_period=params.save_period,
                plots=params.plots,
                project=self.project_name,
                name=self.run_name,
                exist_ok=params.exist_ok
            )
            
            self.logger.info("Training completed successfully")
            return results
            
        except UltralyticsError as e:
            self.logger.error(f"Ultralytics error during training: {e}")
            raise
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during training: {e}")
            raise

# ==========================================================
# TRAINING PIPELINE
# ==========================================================

class TrainingPipeline:
    """
    Orchestrates the complete training pipeline.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the TrainingPipeline.
        
        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger or logging.getLogger("YOLOTraining")
        self.config_manager = ConfigurationManager(logger=self.logger)
        self.dataset_manager = DatasetManager(self.config_manager, logger=self.logger)
        self.trainer: Optional[YOLOTrainer] = None
        
    def run(self) -> None:
        """
        Execute the complete training pipeline.
        """
        start_datetime = datetime.now()
        start_time = time.time()
        
        try:
            self._print_header()
            
            # Load configuration
            self.logger.info("Loading configuration...")
            config = self.config_manager.load()
            
            # Prepare dataset
            self.logger.info("Preparing dataset...")
            dataset_path = self.dataset_manager.prepare_dataset()
            data_yaml = str(dataset_path / "data.yaml")
            
            # Get training configuration
            training_config = self.config_manager.get_training_config()
            model_name = training_config.get("model", "yolo12l.pt")
            project_name = training_config.get("project", "YOLO12_Training")
            run_name = training_config.get("run_name", "Aircraft_v12l")
            
            # Initialize trainer
            self.trainer = YOLOTrainer(
                model_name=model_name,
                project_name=project_name,
                run_name=run_name,
                logger=self.logger
            )
            
            # Load model
            self.trainer.load_model()
            
            # Display training info
            self._print_training_info(config, dataset_path, data_yaml)
            
            # Train model
            self.trainer.train(data_yaml=data_yaml)
            
            # Calculate and display elapsed time
            elapsed = time.time() - start_time
            self._print_completion(start_datetime, elapsed)
            
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            self.logger.warning(f"\nTraining interrupted after {self._format_time(elapsed)}")
            sys.exit(1)
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"\nTraining failed after {self._format_time(elapsed)}")
            self.logger.error(f"Error: {e}")
            raise
    
    def _print_header(self) -> None:
        """Print the training header."""
        self.logger.info("=" * 70)
        self.logger.info("YOLO12L Training Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"Current Directory: {os.getcwd()}")
        self.logger.info("")
    
    def _print_training_info(
        self,
        config: Dict[str, Any],
        dataset_path: Path,
        data_yaml: str
    ) -> None:
        """Print training information."""
        dataset_config = config.get("dataset", {})
        
        self.logger.info("=" * 70)
        self.logger.info("Training Configuration")
        self.logger.info("=" * 70)
        self.logger.info(f"Repository: {dataset_config.get('repository', 'N/A')}")
        self.logger.info(f"Dataset Path: {dataset_path}")
        self.logger.info(f"Data YAML: {data_yaml}")
        self.logger.info(f"Model: {config.get('training', {}).get('model', 'N/A')}")
        self.logger.info(f"Project: {config.get('training', {}).get('project', 'N/A')}")
        self.logger.info(f"Run Name: {config.get('training', {}).get('run_name', 'N/A')}")
        self.logger.info("=" * 70)
        self.logger.info("Training Started")
        self.logger.info("=" * 70)
    
    def _print_completion(self, start_datetime: datetime, elapsed: float) -> None:
        """Print training completion information."""
        end_datetime = datetime.now()
        formatted_time = self._format_time(elapsed)
        
        self.logger.info("=" * 70)
        self.logger.info("Training Completed Successfully")
        self.logger.info("=" * 70)
        self.logger.info(f"Started: {start_datetime}")
        self.logger.info(f"Ended  : {end_datetime}")
        self.logger.info(f"Duration: {formatted_time}")
        self.logger.info("=" * 70)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main() -> None:
    """Main entry point for the training pipeline."""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Create and run pipeline
        pipeline = TrainingPipeline(logger=logger)
        pipeline.run()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()