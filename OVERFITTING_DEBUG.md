# Overfitting Debugging Guide

## Problem
After training on 19GB recording file, bot shows:
- 100% training accuracy (RED FLAG - overfitting!)
- Validation loss: 0.0008 (suspiciously low)
- Bot only clicks bottom-right corner of screen

## Root Cause Hypotheses

### 1. Target Selection Issue (Tactician)
- **Symptom**: Tactician always selects same target index
- **Why**: Model memorized "always select target #N" instead of learning target selection strategy
- **Fix**: Need more diverse data or target selection augmentation

### 2. Bad Target Positions (Unified Policy)
- **Symptom**: Target objects always in bottom-right corner
- **Why**:
  - All training data from same area of map
  - Or YOLO detector consistently placing objects in bottom-right
- **Fix**: Collect data from different map locations

### 3. Executor Residual Connection Amplification
- **Symptom**: Model learns zero correction, relies entirely on residual
- **Why**: When target is wrong, residual forces click to wrong position
- **Location**: `executor.py` lines 152-162
```python
# RESIDUAL CONNECTION: Adds target position to model output
target_xy = target_info[:, :2].clamp(0.01, 0.99)
target_logits = torch.log(target_xy / (1.0 - target_xy))
action[:, 0] = action[:, 0] + target_logits[:, 0]  # X position
action[:, 1] = action[:, 1] + target_logits[:, 1]  # Y position
```
- **Fix**: Check if model learns meaningful corrections or just outputs zeros

### 4. Data Diversity Issue
- **Symptom**: 100% accuracy = model memorized training data
- **Why**: Not enough variation in scenarios/positions/targets
- **Fix**: Data augmentation or more diverse recording sessions

## Debug Logging Added

### 1. Controller Level ([bot_controller_v2.py:1022-1039](f:\dev\bot\darkorbit_bot\v2\bot_controller_v2.py#L1022-L1039))
```
[DEBUG] Mode: FIGHT | Target #2: (0.734, 0.823) [Enemy_npc_streuner] | Mouse: (0.738, 0.825) | Objects: 5 | Events: kills:2
```
Shows:
- Current mode from Strategist
- Target index and position from Tactician
- Final mouse position from Executor
- Number of tracked objects
- **OCR log events** (kills, damage_taken, rewards, alerts from last 5 seconds)

### 2. Unified Policy Level ([unified.py:269-271](f:\dev\bot\darkorbit_bot\v2\models\unified.py#L269-L271))
```
[UNIFIED-DEBUG] Valid objects: 5 | Target idx: 2 | Target pos: (0.734, 0.823)
```
Shows:
- How many valid objects available
- Which target index Tactician selected
- Target position passed to Executor

### 3. Executor Level ([executor.py:164-171](f:\dev\bot\darkorbit_bot\v2\models\executor.py#L164-L171))
```
[EXECUTOR-DEBUG] Raw model output: (0.502, 0.498) | Target: (0.734, 0.823)
```
Shows:
- Model's raw output BEFORE residual connection
- Target position that will be added

## What to Look For

### Signs of Target Selection Overfitting:
- Target index is ALWAYS the same (e.g., always target #0)
- Target index changes but positions are all similar

### Signs of Position Overfitting:
- Target positions cluster in bottom-right (e.g., x>0.7, y>0.7)
- Positions don't vary much across frames

### Signs of Executor Not Learning:
- Raw model output is always ~(0.5, 0.5) (center)
- Raw output doesn't vary based on situation
- Model relies 100% on residual connection

### Good Behavior Would Look Like:
- Target indices vary based on what's on screen
- Target positions spread across screen
- Raw model output shows learned corrections (not always 0.5)

## Running the Test

```bash
# Quick 5-second test
python test_overfitting.py

# Or run manually
python -m darkorbit_bot.v2.bot_controller_v2 --policy-dir v2/models
```

Debug output prints every 30 frames (~0.5 seconds).

## Expected Diagnosis

Based on 100% accuracy, most likely issues are:

1. **Data diversity** - All recordings from same scenario/position
2. **Target selection memorization** - Model learned "always pick target #N"
3. **Executor not learning** - Relies entirely on residual, doesn't learn corrections

## Potential Fixes

### If Data Diversity Issue:
- Record from multiple map locations
- Record different combat scenarios (flee, loot, explore)
- Add position augmentation during training

### If Target Selection Issue:
- Add target selection augmentation (shuffle object order)
- Use curriculum learning (start with obvious targets)
- Check if human player varies target selection

### If Executor Issue:
- Reduce residual connection strength
- Add noise to target positions during training
- Verify training loss actually decreased (not just validation)

## Bug Fixes Applied

### 1. Mouse Position Clamping ([bot_controller_v2.py:1499-1502](f:\dev\bot\darkorbit_bot\v2\bot_controller_v2.py#L1499-L1502))
**Problem**: When cursor goes to another screen during recording, mouse position becomes invalid (negative or >1.0)
**Fix**: Clamp mouse position to valid range [0, 1] before calculating movement
```python
current_x = np.clip(current_x, 0.0, 1.0)
current_y = np.clip(current_y, 0.0, 1.0)
```
This prevents corrupted training data from multi-monitor setups!

### 2. OCR Log Event Tracking ([bot_controller_v2.py:927-929](f:\dev\bot\darkorbit_bot\v2\bot_controller_v2.py#L927-L929))
**Added**: Real-time combat log reading via OCR
- Reads top-screen fast logs every 100ms
- Tracks kills, damage_taken, rewards, alerts
- Events decay after 5 seconds
- Displayed in debug output

## Next Steps

1. Run `test_overfitting.py` to capture debug output
2. Look for patterns in target selection and positions
3. Check if raw model output varies or stays at 0.5
4. **Check if OCR events show up** (if pytesseract installed)
5. Decide which fix to apply based on diagnosis
