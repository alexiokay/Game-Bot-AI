# Data Folders Guide - V1 vs V2

**Date**: 2026-01-23

---

## Summary: Which Folder to Use?

| Bot Version | Training Data Location | Size | Format |
|-------------|------------------------|------|--------|
| **V1 (Reasoning)** | `darkorbit_bot/data/recordings/` | 583 MB | Unknown format |
| **V2 (Hierarchical)** | `darkorbit_bot/data/recordings_v2/` | 539 MB | JSON sequences ✅ |
| **V2 (Shadow)** | `data/recordings/` | **19 GB** | PKL (shadow learning) ✅ |

---

## V2 Training Data (RECOMMENDED)

### Option 1: Shadow Recording (19 GB) ⭐ BEST

**Location**: `data/recordings/shadow_recording_20260123_000916.pkl`

**Size**: 19 GB (huge!)

**What it is**: Shadow learning recording from Jan 23, 2026 at 00:09

**When to use**:
- Training V2 models (Strategist, Tactician, Executor)
- You have TONS of data here
- This is your best training data!

**Training command**:
```bash
python darkorbit_bot/v2/training/train_strategist.py \
    --data data/recordings \
    --output models/v2/strategist/best_model.pt \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```

---

### Option 2: JSON Sequences (539 MB)

**Location**: `darkorbit_bot/data/recordings_v2/`

**Size**: 539 MB

**Format**: JSON sequence files
```
session_20260121_164012/
  sequence_0000_KILL_AGGRESSIVE.json
  sequence_0001_KILL_AGGRESSIVE.json
  sequence_0002_KILL_AGGRESSIVE.json
  ...
```

**When to use**:
- Alternative to shadow recording
- Structured recordings from Jan 21, 2026
- All labeled as "KILL_AGGRESSIVE" mode

**Training command**:
```bash
python darkorbit_bot/v2/training/train_strategist.py \
    --data darkorbit_bot/data/recordings_v2 \
    --output models/v2/strategist/best_model.pt \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```

---

## V1 Training Data

### V1 Recordings (583 MB)

**Location**: `darkorbit_bot/data/recordings/`

**Size**: 583 MB

**For**: V1 reasoning bot training

**When to use**:
- Training V1 bot (not V2!)
- Legacy recordings

---

## Other Important Folders

### VLM Corrections (for fine-tuning)

| Folder | Size | For | Purpose |
|--------|------|-----|---------|
| `darkorbit_bot/data/vlm_corrections/` | 2.8 MB | V1 | VLM corrections for V1 |
| `darkorbit_bot/data/vlm_corrections_v2/` | 24 MB | V2 | VLM corrections for V2 ✅ |
| `darkorbit_bot/data/v2_corrections/` | 14 MB | V2 | Alternative V2 corrections |

**Use for**: Fine-tuning models after initial training with VLM feedback

---

### YOLO Training Data

**Location**: `darkorbit_bot/data/yolo_dataset/`

**Size**: 444 MB

**For**: YOLO object detection model training (not bot AI)

---

### Checkpoints

| Folder | Size | For |
|--------|------|-----|
| `data/checkpoints/` | 0 bytes | Empty |
| `darkorbit_bot/data/checkpoints/` | 20 MB | Old checkpoints |

---

## Recommended Training Workflow for V2

### Step 1: Use Shadow Recording (19 GB)

This is your **BEST data** - 19 GB from shadow learning!

```bash
# Train Strategist (goal selection)
python darkorbit_bot/v2/training/train_strategist.py \
    --data data/recordings \
    --output models/v2/strategist/best_model.pt \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda

# Train Tactician (target selection)
python darkorbit_bot/v2/training/train_tactician.py \
    --data data/recordings \
    --output models/v2/tactician/best_model.pt \
    --epochs 80 \
    --batch-size 64 \
    --lr 1e-4 \
    --device cuda

# Train Executor (precise actions)
python darkorbit_bot/v2/training/train_executor.py \
    --data data/recordings \
    --output models/v2/executor/best_model.pt \
    --epochs 60 \
    --batch-size 128 \
    --lr 3e-4 \
    --device cuda
```

---

### Step 2: Fine-tune with VLM Corrections (Optional)

After initial training, fine-tune with VLM feedback:

```bash
python darkorbit_bot/v2/training/finetune_with_vlm.py \
    --corrections_dir darkorbit_bot/data/vlm_corrections_v2 \
    --model_path models/v2/strategist/best_model.pt \
    --output_dir models/v2/strategist_vlm_ft \
    --epochs 20 \
    --lr 1e-5 \
    --device cuda
```

---

## Quick Reference

### Training V2 Models

**Use this data**: `data/recordings/` (19 GB shadow recording)

**Don't use**:
- ❌ `darkorbit_bot/data/recordings/` (V1 data)
- ❌ `recordings/v2_training` (doesn't exist)

### Training V1 Models

**Use this data**: `darkorbit_bot/data/recordings/` (583 MB)

---

## File Structure Overview

```
project/
├── data/                                  # Root data folder
│   ├── recordings/                        # ⭐ V2 SHADOW DATA (19 GB)
│   │   └── shadow_recording_20260123_000916.pkl
│   └── checkpoints/                       # Empty
│
├── darkorbit_bot/data/                    # Main data folder
│   ├── recordings/                        # V1 recordings (583 MB)
│   ├── recordings_v2/                     # V2 JSON sequences (539 MB)
│   │   └── session_20260121_164012/
│   │       ├── sequence_0000_KILL_AGGRESSIVE.json
│   │       ├── sequence_0001_KILL_AGGRESSIVE.json
│   │       └── ...
│   ├── vlm_corrections/                   # V1 VLM corrections (2.8 MB)
│   ├── vlm_corrections_v2/                # V2 VLM corrections (24 MB)
│   ├── v2_corrections/                    # V2 corrections alt (14 MB)
│   ├── yolo_dataset/                      # YOLO training data (444 MB)
│   ├── checkpoints/                       # Old checkpoints (20 MB)
│   ├── bootstrap/                         # Bootstrap data (12 MB)
│   ├── grounded/                          # Grounded data (5.2 MB)
│   ├── corrections/                       # General corrections (99 MB)
│   └── meta_analysis/                     # Meta analysis (32 KB)
│
└── yolo/datasets/                         # YOLO datasets (separate)
    └── darkorbit_v6/
```

---

## Key Takeaways

1. **For V2 training**: Use `data/recordings/` (19 GB shadow recording) ⭐
2. **For V1 training**: Use `darkorbit_bot/data/recordings/` (583 MB)
3. **For VLM fine-tuning (V2)**: Use `darkorbit_bot/data/vlm_corrections_v2/` (24 MB)
4. **JSON sequences**: `darkorbit_bot/data/recordings_v2/` is alternative V2 data (539 MB)

The 19 GB shadow recording is your goldmine - use it for training V2!
