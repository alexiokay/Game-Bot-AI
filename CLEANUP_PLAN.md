# Data Cleanup Plan

**Date**: 2026-01-23

---

## What to Keep vs Delete

### ✅ KEEP (Important)

| Folder | Size | Why |
|--------|------|-----|
| `darkorbit_bot/data/recordings_v2/` | 539 MB | **V2 training data** - USE THIS! |
| `darkorbit_bot/data/vlm_corrections_v2/` | 24 MB | V2 VLM corrections for fine-tuning |
| `yolo/datasets/` | 98 MB | **REAL YOLO datasets** (darkorbit_v6, etc.) |
| `darkorbit_bot/data/recordings/` | 583 MB | V1 training data (if you use V1) |

**Total to keep**: ~1.2 GB

---

### 🗑️ DELETE (Safe to Remove)

| Folder | Size | Why Delete |
|--------|------|------------|
| `data/recordings/` | 0 MB | Empty folder |
| `data/checkpoints/` | 0 MB | Empty folder |
| `darkorbit_bot/data/auto_labeled/` | 0 MB | Empty |
| `darkorbit_bot/data/lm_studio_labels/` | 0 MB | Empty |
| `darkorbit_bot/data/checkpoints/` | 20 MB | Old checkpoints (outdated) |
| `darkorbit_bot/data/bootstrap/` | 12 MB | Bootstrap data (not needed) |
| `darkorbit_bot/data/grounded/` | 5.2 MB | Grounded data (not needed) |
| `darkorbit_bot/data/corrections/` | 99 MB | Generic corrections (redundant) |
| `darkorbit_bot/data/meta_analysis/` | 32 KB | Old analysis |
| `darkorbit_bot/data/yolo_dataset/` | 444 MB | **OLD/DUPLICATE YOLO data** (real data in yolo/datasets/) |

**Total to delete**: ~580 MB

---

### ❓ MAYBE KEEP

| Folder | Size | Decision |
|--------|------|----------|
| `darkorbit_bot/data/v2_corrections/` | 15 MB | If different from vlm_corrections_v2, keep |
| `darkorbit_bot/data/vlm_corrections/` | 2.8 MB | Only if you use V1 |

---

## Cleanup Commands

### Step 1: Delete Empty/Useless Folders

```bash
# Delete empty folders
rm -rf data/recordings
rm -rf data/checkpoints
rm -rf darkorbit_bot/data/auto_labeled
rm -rf darkorbit_bot/data/lm_studio_labels

# Delete old/unused data
rm -rf darkorbit_bot/data/checkpoints
rm -rf darkorbit_bot/data/bootstrap
rm -rf darkorbit_bot/data/grounded
rm -rf darkorbit_bot/data/corrections
rm -rf darkorbit_bot/data/meta_analysis
rm -rf darkorbit_bot/data/yolo_dataset  # OLD duplicate, real data in yolo/datasets/
```

**Saves**: ~580 MB

---

### Step 2: Check V2 Corrections (Optional)

```bash
# Compare the two V2 correction folders
diff -r darkorbit_bot/data/v2_corrections darkorbit_bot/data/vlm_corrections_v2

# If they're the same, delete v2_corrections
rm -rf darkorbit_bot/data/v2_corrections
```

---

## After Cleanup Structure

```
darkorbit_bot/data/
├── recordings_v2/           # 539 MB - V2 TRAINING DATA ✅
├── vlm_corrections_v2/      # 24 MB  - V2 VLM corrections ✅
├── recordings/              # 583 MB - V1 data (optional)
└── vlm_corrections/         # 2.8 MB - V1 corrections (optional)

yolo/datasets/
├── darkorbit_v6/            # 27 MB  - YOLO v6 dataset ✅
├── darkorbit_v7_simple/     # 27 MB  - YOLO v7 dataset ✅
└── ...                      # Other YOLO versions
```

**Total**: ~1.2 GB (clean!)

---

## Updated Training Commands (NOW WITH DEFAULTS!)

### Simple Commands (Use Defaults)

```bash
# Strategist - just run it!
python darkorbit_bot/v2/training/train_strategist.py --epochs 100 --batch-size 32 --lr 1e-4

# Tactician
python darkorbit_bot/v2/training/train_tactician.py --epochs 80 --batch-size 64 --lr 1e-4

# Executor
python darkorbit_bot/v2/training/train_executor.py --epochs 60 --batch-size 128 --lr 3e-4
```

**Defaults now automatically use**:
- Data: `darkorbit_bot/data/recordings_v2/` ✅
- Output: `models/v2/strategist/best_model.pt` ✅
- Device: `cuda` ✅

---

### Custom Data Path (If You Want Different Folder)

```bash
# Use custom recordings folder
python darkorbit_bot/v2/training/train_strategist.py \
    --data path/to/other/recordings \
    --epochs 100 --batch-size 32 --lr 1e-4
```

---

## Shadow Recording Fixed

When recording new shadow data:

```bash
# Correct way (saves to data/recordings/ - but we should fix this!)
python -m darkorbit_bot.v2.main --shadow --save-recordings
```

**TODO**: Update bot_controller_v2.py to save to better location:
- Current: `data/recordings/`
- Should be: `darkorbit_bot/data/recordings_v2/shadow/`

---

## One-Line Cleanup Script

```bash
# Delete all unnecessary folders at once
rm -rf data/recordings data/checkpoints darkorbit_bot/data/{auto_labeled,lm_studio_labels,checkpoints,bootstrap,grounded,corrections,meta_analysis} && echo "Cleaned up! Saved ~136 MB"
```

---

## Verification After Cleanup

```bash
# Check what's left
du -sh darkorbit_bot/data/*

# Should show:
# 539M  darkorbit_bot/data/recordings_v2      ✅ V2 training data
# 24M   darkorbit_bot/data/vlm_corrections_v2 ✅ V2 corrections
# 444M  darkorbit_bot/data/yolo_dataset       ✅ YOLO data
# 583M  darkorbit_bot/data/recordings         ✅ V1 data (optional)
```

---

## Summary

### What Changed:

1. **Training scripts now have defaults** - No need to specify `--data` every time!
2. **Defaults point to correct V2 data**: `darkorbit_bot/data/recordings_v2/`
3. **Output paths are sensible**: `models/v2/strategist/best_model.pt`

### What to Delete:

- Empty folders (0 MB)
- Old checkpoints (20 MB)
- Bootstrap/grounded/corrections (116 MB)
- **Total savings**: ~136 MB

### Simple Training Now:

```bash
# Just run with minimal args!
python darkorbit_bot/v2/training/train_strategist.py --epochs 100 --batch-size 32
python darkorbit_bot/v2/training/train_tactician.py --epochs 80 --batch-size 64
python darkorbit_bot/v2/training/train_executor.py --epochs 60 --batch-size 128
```

✅ Training commands now work with correct defaults!
✅ No need to remember which folder to use!
✅ Clean data structure!
