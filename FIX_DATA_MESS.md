# Fix Data Mess - Complete Guide

**Date**: 2026-01-23

---

## Current Problems

### 1. No Recent Shadow Recordings
- You ran shadow training but no files were saved
- `data/recordings/` is **EMPTY**
- Old recordings from Jan 21 are in `darkorbit_bot/data/recordings_v2/`

### 2. Resolution Confusion
- Bot auto-detects your monitor resolution
- If you're on 1440p (2560x1440), it should use that
- If on 1080p (1920x1080), it uses that
- **The resolution is correct** - it matches your monitor!

### 3. Folder Mess
- Multiple `data` folders confusing
- V1 and V2 data mixed together
- No clear structure

---

## Why Shadow Recording Didn't Save

Shadow trainer only saves when:
1. You use `--save-recordings` flag
2. You properly exit the bot (Ctrl+C gracefully)

**Check if you used the flag**:
```bash
# WRONG (no recordings saved):
python -m darkorbit_bot.v2.main --shadow

# CORRECT (saves recordings):
python -m darkorbit_bot.v2.main --shadow --save-recordings
```

---

## Fix #1: How to Use Shadow Training Correctly

### Step 1: Start shadow mode WITH save flag

```bash
python -m darkorbit_bot.v2.main --shadow --save-recordings
```

### Step 2: Play the game normally
- Bot watches your mouse/keyboard
- No automation - you're playing manually
- Bot learns from your actions

### Step 3: Exit GRACEFULLY (important!)
- Press `Ctrl+C` ONCE
- Wait for "Saving recordings..." message
- Don't force-kill or close terminal

**Output should show**:
```
[SHADOW] Saving 5000 demonstrations...
[SHADOW] Compressing frames... (this may take a minute)
[SHADOW] ✓ Saved to data/recordings/shadow_recording_TIMESTAMP.pkl (500 MB)
```

---

## Fix #2: Check Monitor Resolution

Run this to see what resolution the bot detects:

```bash
python -m darkorbit_bot.v2.main --test
```

Look for this line:
```
Screen: 2560x1440 at (0, 0)    # 1440p ✓
# OR
Screen: 1920x1080 at (0, 0)    # 1080p
```

**The resolution is auto-detected and CORRECT!**

If you have multiple monitors:
```bash
# Use monitor 1 (primary)
python -m darkorbit_bot.v2.main --monitor 1

# Use monitor 2 (secondary, if that's where game is)
python -m darkorbit_bot.v2.main --monitor 2
```

---

## Fix #3: Clean Up Folder Structure

### Current Mess

```
project/
├── data/
│   ├── recordings/              # EMPTY (no shadow recordings saved)
│   └── checkpoints/             # EMPTY
│
└── darkorbit_bot/data/
    ├── recordings/              # 583 MB (V1 data)
    ├── recordings_v2/           # 539 MB (V2 data from Jan 21)
    ├── vlm_corrections/         # 2.8 MB (V1)
    ├── vlm_corrections_v2/      # 24 MB (V2)
    ├── v2_corrections/          # 14 MB (V2 alt)
    ├── checkpoints/             # 20 MB (old)
    └── yolo_dataset/            # 444 MB (YOLO)
```

### Proposed Clean Structure

```
project/
├── training_data/               # All training data here
│   ├── v1/                      # V1 bot data
│   │   ├── recordings/          # V1 gameplay
│   │   └── vlm_corrections/     # V1 VLM feedback
│   │
│   ├── v2/                      # V2 bot data
│   │   ├── recordings/          # V2 gameplay (normal recording)
│   │   ├── shadow/              # V2 shadow recordings
│   │   └── vlm_corrections/     # V2 VLM feedback
│   │
│   └── yolo/                    # YOLO detection
│       └── datasets/            # YOLO labeled images
│
├── models/                      # Trained models
│   ├── v1/                      # V1 model checkpoints
│   ├── v2/                      # V2 model checkpoints
│   │   ├── strategist/
│   │   ├── tactician/
│   │   └── executor/
│   └── yolo/                    # YOLO detection models
│
└── logs/                        # Training logs
    ├── v1/
    └── v2/
```

---

## Reorganization Script

Create `reorganize_data.sh`:

```bash
#!/bin/bash

# Create new structure
mkdir -p training_data/v1/recordings
mkdir -p training_data/v1/vlm_corrections
mkdir -p training_data/v2/recordings
mkdir -p training_data/v2/shadow
mkdir -p training_data/v2/vlm_corrections
mkdir -p training_data/yolo/datasets

# Move V1 data
echo "Moving V1 data..."
cp -r darkorbit_bot/data/recordings/* training_data/v1/recordings/ 2>/dev/null
cp -r darkorbit_bot/data/vlm_corrections/* training_data/v1/vlm_corrections/ 2>/dev/null

# Move V2 data
echo "Moving V2 data..."
cp -r darkorbit_bot/data/recordings_v2/* training_data/v2/recordings/ 2>/dev/null
cp -r darkorbit_bot/data/vlm_corrections_v2/* training_data/v2/vlm_corrections/ 2>/dev/null
cp -r darkorbit_bot/data/v2_corrections/* training_data/v2/vlm_corrections/ 2>/dev/null

# Move YOLO data
echo "Moving YOLO data..."
cp -r darkorbit_bot/data/yolo_dataset/* training_data/yolo/datasets/ 2>/dev/null

echo "Done! New structure created in training_data/"
echo "Old data left in place (not deleted)"
```

Make it executable and run:
```bash
chmod +x reorganize_data.sh
./reorganize_data.sh
```

---

## Fix #4: Update Bot to Use New Paths

After reorganizing, update the bot code:

### In `bot_controller_v2.py` line 424:

**Change from**:
```python
recording_dir="data/recordings"
```

**Change to**:
```python
recording_dir="training_data/v2/shadow"
```

### In training scripts:

**Strategist**:
```bash
python darkorbit_bot/v2/training/train_strategist.py \
    --data training_data/v2/recordings \
    --output models/v2/strategist/best_model.pt \
    --epochs 100 --batch-size 32 --lr 1e-4 --device cuda
```

**Tactician**:
```bash
python darkorbit_bot/v2/training/train_tactician.py \
    --data training_data/v2/recordings \
    --output models/v2/tactician/best_model.pt \
    --epochs 80 --batch-size 64 --lr 1e-4 --device cuda
```

**Executor**:
```bash
python darkorbit_bot/v2/training/train_executor.py \
    --data training_data/v2/recordings \
    --output models/v2/executor/best_model.pt \
    --epochs 60 --batch-size 128 --lr 3e-4 --device cuda
```

---

## Quick Fix: Use Existing V2 Data

**Don't reorganize yet!** Just use what you have:

```bash
# Train with existing V2 recordings (Jan 21)
python darkorbit_bot/v2/training/train_strategist.py \
    --data darkorbit_bot/data/recordings_v2 \
    --output models/v2/strategist/best_model.pt \
    --epochs 100 --batch-size 32 --lr 1e-4 --device cuda
```

This will work fine! The recordings from Jan 21 are good V2 training data.

---

## Summary

### Problem 1: No Shadow Recordings
**Solution**: Use `--save-recordings` flag and exit gracefully:
```bash
python -m darkorbit_bot.v2.main --shadow --save-recordings
# Play game...
# Press Ctrl+C ONCE and wait for save
```

### Problem 2: Resolution
**Solution**: No problem! Bot auto-detects your 1440p monitor correctly.

### Problem 3: Folder Mess
**Solution**: For now, use existing data:
```bash
--data darkorbit_bot/data/recordings_v2
```

Later, reorganize with the script above.

---

## Next Steps

1. **Record new shadow data** (with `--save-recordings` flag)
2. **Train with existing V2 data** (darkorbit_bot/data/recordings_v2)
3. **Reorganize later** when you have time

The existing V2 recordings (539 MB) are perfect for training - use those!
