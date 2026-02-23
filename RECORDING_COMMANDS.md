# V2 Recording Commands - Quick Reference

**Date**: 2026-01-23

---

## For Fresh Recording (Recommended)

### Shadow Training with Recordings

```bash
# Using uv (recommended)
uv run darkorbit_bot/v2/bot_controller_v2.py --shadow-train --save-recordings

# Or regular python (if venv activated)
python darkorbit_bot/v2/bot_controller_v2.py --shadow-train --save-recordings
```

**What happens**:
- Bot watches you play (passive, doesn't control)
- Learns in real-time
- Saves recordings to `data/recordings/` for offline training
- Press F1 to start/pause recording
- Mouse cursor locked to screen (prevents bad training data)

**IMPORTANT - Stopping Without Losing Data:**
- Press Ctrl+C **ONCE** and wait
- Bot will save demonstrations (should be instant, ~1-2 seconds)
- Wait for "✓ Saved" message before closing
- If it hangs for more than 10 seconds, force-kill with: `taskkill //F //IM python.exe`

**Output**: `data/recordings/shadow_recording_TIMESTAMP.pkl`

---

## Alternative: Pure Recording (Manual Control)

If you want manual control over what gets saved:

```bash
# Using uv
uv run -m darkorbit_bot.v2.recording.recorder_v2 --model F:/dev/bot/best.pt --monitor 1

# Or regular python
python -m darkorbit_bot.v2.recording.recorder_v2 --model F:/dev/bot/best.pt --monitor 1
```

**Hotkeys**:
- F5 = Start/stop recording
- F6 = Save as "good"
- F7 = Discard buffer

**Features**:
- Mouse cursor locked to screen (prevents bad training data)

**Output**: `recordings/v2_TIMESTAMP/sequence_*.json`

---

## Key Differences

| Feature | Shadow Training | Pure Recording |
|---------|----------------|----------------|
| **Real-time learning** | ✅ Yes | ❌ No |
| **Saves recordings** | ✅ Yes (with --save-recordings) | ✅ Yes |
| **Manual control** | F1 start/pause | F5/F6/F7 |
| **Bot interference** | None (passive) | None |
| **Mouse cursor lock** | ✅ Yes | ✅ Yes |
| **Best for** | Training from scratch | Clean demos |

---

## Common Issues

### Issue: "ImportError: attempted relative import with no known parent package"

**Wrong**: `python darkorbit_bot/v2/recording/recorder_v2.py` (direct script)

**Correct**: `python -m darkorbit_bot.v2.recording.recorder_v2` (as module)

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Activate the correct virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Or use uv
uv run darkorbit_bot/v2/bot_controller_v2.py --shadow-train --save-recordings
```

---

## Recommended Workflow for Fresh Start

Since you want to delete all old recordings and start fresh:

### Step 1: Delete Old Recordings

```bash
# Delete all old recordings (V1 and V2)
rm -rf darkorbit_bot/data/recordings
rm -rf darkorbit_bot/data/recordings_v2
rm -rf data/recordings

# Also clean up duplicate/old data
rm -rf darkorbit_bot/data/yolo_dataset
rm -rf darkorbit_bot/data/checkpoints
rm -rf darkorbit_bot/data/corrections
rm -rf darkorbit_bot/data/bootstrap
rm -rf darkorbit_bot/data/grounded
rm -rf darkorbit_bot/data/meta_analysis
```

**Space freed**: ~1.2 GB

### Step 2: Record Fresh Data (1 hour)

```bash
uv run darkorbit_bot/v2/bot_controller_v2.py --shadow-train --save-recordings
```

Play for 1 hour showing all behaviors:
- Fight NPCs
- Collect loot
- Flee from danger
- Explore maps
- Cautious play

### Step 3: Train V2 Models (4-6 hours)

```bash
# Train Strategist
uv run darkorbit_bot/v2/training/train_strategist.py --epochs 100 --batch-size 32

# Train Tactician
uv run darkorbit_bot/v2/training/train_tactician.py --epochs 80 --batch-size 64

# Train Executor
uv run darkorbit_bot/v2/training/train_executor.py --epochs 60 --batch-size 128
```

**Note**: Training scripts now use correct defaults automatically!
- Data: `darkorbit_bot/data/recordings_v2/`
- Output: `models/v2/{model}/best_model.pt`

---

## Summary

**Use this command for fresh recording**:
```bash
# With uv
uv run darkorbit_bot/v2/bot_controller_v2.py --shadow-train --save-recordings

# Or regular python
python darkorbit_bot/v2/bot_controller_v2.py --shadow-train --save-recordings
```

This is the simplest and most effective way to record training data for V2!
