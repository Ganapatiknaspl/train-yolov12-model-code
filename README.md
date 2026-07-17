# YOLO12L Aircraft Detection — Training with Auto GitHub Sync

Trains a **YOLOv12L** model on aerial object detection (airplane, drone, helicopter, bird) and **automatically pushes all training outputs to GitHub** after every checkpoint — so nothing is lost even if the cloud session is interrupted.

---

## Classes

| ID | Class |
|----|-----------|
| 0 | Airplane |
| 1 | Drone |
| 2 | Helicopter |
| 3 | Bird |

---

## How It Works

```
python model.py
```

Running `model.py` does **everything automatically**:

1. **Clones the YOLO config** from the code repo (`YOLO_Config/`)
2. **Clones or updates the dataset** from the dataset repo (`AerialDataset/`)
3. **Validates the dataset** — removes orphaned labels
4. **Starts training** — YOLOv12L with the configured hyperparameters
5. **Pushes to GitHub every 10 epochs** — epoch10.pt, epoch20.pt, plots, metrics
6. **Pushes on training completion** — final best.pt, last.pt, args.yaml, everything
7. **Pushes on interruption** — if lightning.ai kills the job mid-run, an emergency push fires automatically

All training outputs are stored in the dataset GitHub repo under:
```
runs/
  YOLO12_Training/
    Aircraft_v12l/
      weights/
        best.pt
        last.pt
        epoch10.pt
        epoch20.pt
        ...
      args.yaml
      results.csv
      confusion_matrix.png
      PR_curve.png
      F1_curve.png
      ...
```

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | Main training script — run this |
| `github_sync.py` | GitHub push engine (imported by model.py) |
| `requirements.txt` | Python dependencies |
| `.env` | **Your local token file** — create this yourself, never commit it |

---

## Setup: GitHub Personal Access Token (PAT)

You need a GitHub PAT with **`repo` write access** to push training outputs.

### Create a PAT
1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**
2. Click **Generate new token (classic)**
3. Give it a name e.g. `yolo-training-sync`
4. Under **Scopes** tick: ✅ `repo` (full control of private repositories)
5. Click **Generate token** — copy it now (you won't see it again)

---

## Running on lightning.ai ☁️

### Step 1 — Add the token as a secret
1. Open your lightning.ai Studio
2. Go to **Settings → Secrets**
3. Click **Add Secret**
   - **Name**: `GITHUB_TOKEN`
   - **Value**: `ghp_xxxxxxxxxxxxxxxxxxxx` (your PAT)
4. Click Save

> ✅ That's it. The script reads it automatically from the environment.
> You do NOT need to change any code.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run
```bash
python model.py
```

### Step 4 — Watch outputs appear on GitHub
Open your dataset repo → `runs/YOLO12_Training/Aircraft_v12l/` — you'll see folders and files being committed every 10 epochs.

---

## Running Locally 💻

### Step 1 — Create a `.env` file
In the same folder as `model.py`, create a file called `.env`:
```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```
> ⚠️ Never commit `.env` to git. Add it to `.gitignore`.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run
```bash
python model.py
```

> On Windows (if you prefer environment variable over .env):
> ```
> set GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
> python model.py
> ```

---

## Dataset

The dataset is automatically cloned from:
```
https://github.com/Ganapatiknaspl/train-yolo12l-model
```

Structure expected in the repo:
```
AerialDataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv12L |
| Image Size | 1024 |
| Batch Size | 16 |
| Epochs | 100 |
| Patience | 20 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.001 |
| Mixed Precision | ✅ Enabled |
| Seed | 42 |
| Save Period | Every 10 epochs |
| GitHub Sync | Every 10 epochs + on exit |

---

## Data Augmentation

- Mosaic (1.0)
- MixUp (0.15)
- Copy Paste (0.30)
- HSV Augmentation
- Random Scale (0.50)
- Translation (0.10)
- Horizontal Flip (0.50)
- Vertical Flip (0.20)
- Shear (2.0)
- Perspective (0.0005)

---

## Git LFS (optional — for very large weight files)

GitHub blocks single files over **100 MB** without Git LFS.
`yolo12l.pt` weights are ~54 MB so they push fine without LFS.
If your weights ever exceed 100 MB, install LFS:

```bash
# Ubuntu / lightning.ai
sudo apt install git-lfs
git lfs install

# macOS
brew install git-lfs
```

The sync script detects LFS automatically and enables `*.pt` tracking.

---

## Citation

```
@software{ultralytics,
  author = {Ultralytics},
  title  = {Ultralytics YOLO},
  year   = {2025}
}
```
