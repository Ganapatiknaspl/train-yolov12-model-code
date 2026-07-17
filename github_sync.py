"""
github_sync.py
==============
Utility module that pushes YOLO training run outputs to a GitHub repository
automatically during and after training.

HOW IT WORKS
------------
  1. At startup: clones the target GitHub repo into a local `runs_repo/` folder.
  2. Every 10 epochs: copies whatever YOLO has saved so far into the cloned
     repo folder and pushes a commit to GitHub.
  3. At training end: pushes the complete final run (best.pt, all plots, etc.).
  4. On any crash / interruption: an atexit + SIGTERM handler triggers one last
     push so nothing is lost even if lightning.ai kills the job mid-epoch.

WHAT GETS PUSHED
----------------
  Everything inside the YOLO run folder, including:
    weights/best.pt        – best weights ever
    weights/last.pt        – last checkpoint
    weights/epoch10.pt     – periodic intermediate checkpoints
    weights/epoch20.pt     ...
    args.yaml              – training hyperparameters
    results.csv            – per-epoch metrics
    confusion_matrix.png   – and all other plots

AUTHENTICATION (REQUIRED)
--------------------------
  You need a GitHub Personal Access Token (PAT) with repo write access.

  ── Option A: Environment variable (recommended for lightning.ai) ──
    On lightning.ai:
      Go to → Settings → Secrets → Add Secret
      Name : GITHUB_TOKEN
      Value: ghp_xxxxxxxxxxxxxxxxxxxx
    That's it. The script reads it automatically.

    On any other cloud / server:
      export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

  ── Option B: .env file (recommended for local testing) ──
    Create a file called `.env` in the same folder as model.py:
      GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
    The script loads it automatically via python-dotenv.
    (Make sure .env is listed in .gitignore — never commit your token!)

GIT LFS (optional but recommended for *.pt files > 100 MB)
----------------------------------------------------------
  GitHub blocks single files > 100 MB without Git LFS.
  Install: sudo apt install git-lfs   (on Ubuntu / lightning.ai)
           brew install git-lfs        (on macOS)
           choco install git-lfs       (on Windows)
  If LFS is installed, this script enables it automatically.
  If not installed, files under 100 MB still push fine (last.pt / best.pt
  for yolo12l are ~54 MB so they're within the limit without LFS).
"""

from __future__ import annotations   # enables list[str] on Python 3.8 / 3.9

import os
import re
import shutil
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Resolve GitHub token from environment or .env file
# ──────────────────────────────────────────────────────────────────────────────

def _load_token() -> str:
    """
    Returns the GitHub PAT from:
      1. GITHUB_TOKEN environment variable (set by lightning.ai secrets, or
         manually with `export GITHUB_TOKEN=...`)
      2. A .env file in the current directory (for local development)
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return token

    # Try loading from a .env file (python-dotenv)
    env_file = Path(".env")
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(env_file)
            token = os.environ.get("GITHUB_TOKEN", "")
            if token:
                logger.info("Loaded GITHUB_TOKEN from .env file.")
                return token
        except ImportError:
            pass  # dotenv not installed, continue without it

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, logging stdout + stderr."""
    logger.info("  $ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout.strip():
        logger.info("  stdout: %s", result.stdout.strip())
    if result.stderr.strip():
        logger.warning("  stderr: %s", result.stderr.strip())
    return result


def _inject_token(repo_url: str, token: str) -> str:
    """
    Injects a GitHub PAT into an HTTPS URL so git can push without prompting.
      https://github.com/user/repo.git
    →  https://<token>@github.com/user/repo.git
    """
    # Strip any existing credentials first to avoid duplication
    url = re.sub(r"https://[^@]*@", "https://", repo_url)
    return url.replace("https://", f"https://{token}@", 1)


def _lfs_available() -> bool:
    """Returns True if git-lfs is installed."""
    return shutil.which("git-lfs") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class GitHubSync:
    """
    Pushes YOLO training outputs to a GitHub repository automatically.

    Parameters
    ----------
    repo_url : str
        HTTPS URL of the destination GitHub repo.
        e.g. "https://github.com/Ganapatiknaspl/train-yolo12l-model"
    local_run_dir : str | Path
        Path to the YOLO run directory created during training.
        e.g. "YOLO12_Training/Aircraft_v12l"
    project : str
        YOLO project name  (same as `project=` in model.train()).
    run_name : str
        YOLO run name  (same as `name=` in model.train()).
    token : str, optional
        GitHub PAT. If not provided, read from GITHUB_TOKEN env var or .env.
    local_repo_dir : str | Path
        Local directory where the destination repo is cloned. Default: "runs_repo".
    """

    def __init__(
        self,
        repo_url: str,
        local_run_dir,
        project: str,
        run_name: str,
        token: str = None,
        local_repo_dir = "runs_repo",
    ):
        self.repo_url = repo_url.rstrip("/")
        if not self.repo_url.endswith(".git"):
            self.repo_url += ".git"

        self.local_run_dir   = Path(local_run_dir)
        self.project         = project
        self.run_name        = run_name
        self.local_repo_dir  = Path(local_repo_dir)

        # Resolve token
        self.token = token or _load_token()
        if not self.token:
            raise ValueError(
                "\n"
                "GitHub token not found!\n\n"
                "On lightning.ai:\n"
                "  → Settings → Secrets → Add Secret\n"
                "  Name : GITHUB_TOKEN\n"
                "  Value: ghp_xxxxxxxxxxxxxxxxxxxx\n\n"
                "For local testing — create a .env file next to model.py:\n"
                "  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx\n"
            )

        self.authenticated_url = _inject_token(self.repo_url, self.token)

        # Path inside the cloned repo where the run folder will be stored
        self.dest_in_repo = self.local_repo_dir / "runs" / self.project / self.run_name

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Clone the destination GitHub repo locally (once) or pull latest changes.
        Also configures Git identity and enables LFS for *.pt files if available.
        """
        logger.info("=" * 60)
        logger.info("GitHubSync: Setting up destination repo")
        logger.info("  Repo URL  : %s", self.repo_url)
        logger.info("  Local dir : %s", self.local_repo_dir)
        logger.info("=" * 60)

        if not self.local_repo_dir.exists():
            logger.info("Cloning destination repo (shallow)...")
            _run([
                "git", "clone", "--depth", "1",
                self.authenticated_url,
                str(self.local_repo_dir),
            ])
        else:
            logger.info("Repo already cloned. Updating remote URL and pulling...")
            _run(
                ["git", "remote", "set-url", "origin", self.authenticated_url],
                cwd=str(self.local_repo_dir),
            )
            _run(
                ["git", "pull", "--rebase"],
                cwd=str(self.local_repo_dir),
                check=False,   # don't fail if nothing to pull
            )

        # Configure git identity — required in headless / cloud environments
        _run(["git", "config", "user.email", "training-bot@github-sync.local"],
             cwd=str(self.local_repo_dir))
        _run(["git", "config", "user.name", "YOLO Training Bot"],
             cwd=str(self.local_repo_dir))

        # Enable Git LFS for *.pt weight files if available
        if _lfs_available():
            logger.info("Git LFS detected — tracking *.pt files with LFS.")
            _run(["git", "lfs", "install"], cwd=str(self.local_repo_dir), check=False)
            _run(["git", "lfs", "track", "*.pt"], cwd=str(self.local_repo_dir), check=False)
        else:
            logger.info(
                "Git LFS not installed. *.pt files will push normally "
                "(fine as long as they are under 100 MB each)."
            )

        logger.info("GitHubSync: Destination repo is ready.")

    def push(self, commit_message: str = None):
        """
        Copy the current YOLO run folder into the cloned repo and push to GitHub.

        The full run folder is mirrored — including all epoch*.pt checkpoints,
        best.pt, last.pt, args.yaml, results.csv, and all plots.

        Parameters
        ----------
        commit_message : str, optional
            Custom commit message. Auto-generated if not provided.
        """
        logger.info("=" * 60)
        logger.info("GitHubSync: Pushing to GitHub")
        logger.info("  Source : %s", self.local_run_dir)
        logger.info("  Dest   : %s", self.dest_in_repo)
        logger.info("=" * 60)

        if not self.local_run_dir.exists():
            logger.warning(
                "YOLO run directory '%s' does not exist yet — skipping push.",
                self.local_run_dir,
            )
            return

        # Pull before writing to avoid conflicts
        _run(["git", "pull", "--rebase"], cwd=str(self.local_repo_dir), check=False)

        # ── Mirror the run folder into the repo ───────────────────────────────
        self.dest_in_repo.mkdir(parents=True, exist_ok=True)

        files_copied = 0
        for item in self.local_run_dir.rglob("*"):
            if item.is_file():
                rel       = item.relative_to(self.local_run_dir)
                dest_file = self.dest_in_repo / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_file)
                logger.info("  Copied: %s", rel)
                files_copied += 1

        logger.info("Copied %d file(s) to repo.", files_copied)

        # ── Stage all changes ─────────────────────────────────────────────────
        _run(["git", "add", "-A"], cwd=str(self.local_repo_dir))

        # Check if there's anything new to commit
        status = _run(
            ["git", "status", "--porcelain"],
            cwd=str(self.local_repo_dir),
            check=False,
        )
        if not status.stdout.strip():
            logger.info("GitHubSync: Nothing new to commit — repo already up to date.")
            return

        # ── Commit ────────────────────────────────────────────────────────────
        if not commit_message:
            commit_message = (
                f"Training sync: {self.project}/{self.run_name}"
            )
        _run(["git", "commit", "-m", commit_message], cwd=str(self.local_repo_dir))

        # ── Push ──────────────────────────────────────────────────────────────
        logger.info("Pushing to GitHub...")
        _run(["git", "push", "origin", "main"], cwd=str(self.local_repo_dir))
        logger.info(
            "GitHubSync: Push complete! View at: %s/tree/main/runs/%s/%s",
            self.repo_url.replace(".git", ""),
            self.project,
            self.run_name,
        )

    def push_checkpoint(self, epoch: int):
        """Mid-training push: called every N epochs during training."""
        self.push(
            commit_message=f"[Checkpoint epoch={epoch}] {self.project}/{self.run_name}"
        )

    def push_final(self):
        """End-of-training push: called once after training completes."""
        self.push(
            commit_message=f"[Training complete] {self.project}/{self.run_name}"
        )

    def push_emergency(self):
        """
        Emergency push — called by atexit / SIGTERM handler when training is
        interrupted (e.g. lightning.ai credits exhausted, Ctrl+C, OOM kill).
        Swallows all exceptions so it never crashes the parent process.
        """
        logger.info("GitHubSync: Emergency push triggered (process is exiting)...")
        try:
            self.push(
                commit_message=(
                    f"[EMERGENCY SAVE — interrupted] {self.project}/{self.run_name}"
                )
            )
            logger.info("GitHubSync: Emergency push succeeded.")
        except Exception as exc:
            logger.error("GitHubSync: Emergency push failed: %s", exc)
