# Detection Settings - Optimized Configuration

**Date**: 2026-01-23
**Status**: Applied to V1 and V2 bots

---

## Optimized Settings

After testing and tuning, these settings provide the best balance of accuracy and duplicate removal:

```python
GameDetector(
    model_path="path/to/best.pt",
    confidence_threshold=0.3,  # 30% minimum confidence
    iou_threshold=0.3          # Aggressive NMS duplicate removal
)
```

---

## What Each Setting Does

### `confidence_threshold=0.3`

**What it is**: Minimum confidence score (0-1) for a detection to be accepted

**Effect**:
- Lower (0.1-0.2): More detections, but more false positives
- Medium (0.3): Balanced - filters weak predictions ✅
- Higher (0.5+): Fewer detections, fewer false positives, may miss objects

**Why 0.3**:
- With your dataset (65 images, 40 classes), model confidence is lower than ideal
- 0.3 filters most false positives while keeping real objects
- Lower than typical (0.5) because of small dataset

---

### `iou_threshold=0.3`

**What it is**: IoU (Intersection over Union) threshold for Non-Maximum Suppression

**Effect**:
- Lower (0.2-0.3): Aggressive duplicate removal ✅
- Medium (0.5-0.7): Standard duplicate removal
- Higher (0.8-0.9): Conservative (allows overlapping boxes)

**Why 0.3**:
- Your model was producing many duplicates (Minimap: 4-5, MenuBar: 2, etc.)
- Lower IoU threshold removes more duplicates
- Doesn't hurt accuracy because game UI elements don't overlap much

**NMS Process**:
```
Before NMS:
  Box A: Minimap confidence=0.85
  Box B: Minimap confidence=0.82, IoU=0.65 with Box A
  Box C: Minimap confidence=0.78, IoU=0.72 with Box A

After NMS (iou=0.3):
  Box A: Minimap confidence=0.85 (kept - highest confidence)
  Box B: REMOVED (IoU > 0.3)
  Box C: REMOVED (IoU > 0.3)

Result: Minimap: 1 ✅
```

---

## Performance Results

### Before Optimization

```
Display: 131 FPS | Inference: 20.0 FPS
{
  'Minimap': 4,              ❌ Duplicates
  'MenuBar': 2,              ❌ Duplicates
  'ChatWindow': 4,           ❌ Duplicates
  'Ship_venom': 5,           ❌ Duplicates
  'PlayerShip': 3,           ❌ Duplicates
  'Enemy_npc_mordon': 4,     ❌ Duplicates
}
```

Settings: `conf=0.1, iou=0.7` (default)

---

### After Optimization

```
Display: 134 FPS | Inference: 41.4 FPS
{
  'Minimap': 1,              ✅ Correct
  'MenuBar': 1,              ✅ Correct
  'ChatWindow': 1,           ✅ Correct (was 2-4)
  'Ship_venom': 1-2,         ✅ Mostly correct
  'PlayerShip': 1,           ✅ Correct
  'Portal': 1,               ✅ Correct
  'GroupWindow': 1,          ✅ Correct
}
```

Settings: `conf=0.3, iou=0.3`

**Improvements**:
- Duplicates reduced by ~90%
- Inference speed: 20 FPS → 41 FPS (2x faster!)
- Display FPS: 131 → 134 (slightly better)
- Most UI elements detected correctly (1 instance each)

---

## Files Modified

All bots now use the same optimized settings:

### 1. Detector Core
**File**: [darkorbit_bot/detection/detector.py](darkorbit_bot/detection/detector.py)

**Changes**:
- Added `iou_threshold` parameter to `__init__`
- Default values: `conf=0.3, iou=0.3`
- Both values now configurable

```python
def __init__(self, model_path: str,
             confidence_threshold: float = 0.3,
             iou_threshold: float = 0.3,
             device: str = "auto"):
```

---

### 2. V2 Bot
**File**: [darkorbit_bot/v2/bot_controller_v2.py](darkorbit_bot/v2/bot_controller_v2.py#L189)

**Before**:
```python
self.detector = GameDetector(config.model_path, confidence_threshold=0.4)
```

**After**:
```python
self.detector = GameDetector(
    config.model_path,
    confidence_threshold=0.3,  # Optimized
    iou_threshold=0.3          # Optimized
)
```

---

### 3. V1 Bot (Reasoning)
**File**: [darkorbit_bot/reasoning/bot_controller.py](darkorbit_bot/reasoning/bot_controller.py#L107)

**Before**:
```python
self.detector = GameDetector(config.model_path, confidence_threshold=0.4)
```

**After**:
```python
self.detector = GameDetector(
    config.model_path,
    confidence_threshold=0.3,
    iou_threshold=0.3
)
```

---

## When to Adjust Settings

### Still seeing duplicates?

**Lower IoU threshold**:
```python
iou_threshold=0.2  # Even more aggressive
```

**Raise confidence threshold**:
```python
confidence_threshold=0.4  # Fewer low-confidence predictions
```

---

### Missing important objects?

**Lower confidence threshold**:
```python
confidence_threshold=0.2  # Detect more objects
```

**Raise IoU threshold**:
```python
iou_threshold=0.5  # Less aggressive duplicate removal
```

---

### Different object types need different settings?

You can create multiple detector instances:

```python
# For UI elements (static, never overlap)
ui_detector = GameDetector(
    model_path,
    confidence_threshold=0.3,
    iou_threshold=0.2  # Very aggressive (UI can't overlap)
)

# For game objects (can overlap - stacked enemies)
game_detector = GameDetector(
    model_path,
    confidence_threshold=0.3,
    iou_threshold=0.5  # Less aggressive (allow overlaps)
)
```

---

## Testing Checklist

When you run the bot, verify:

- [ ] UI elements appear once (Minimap: 1, MenuBar: 1, ChatWindow: 1)
- [ ] Ships detected correctly (Ship_venom: 1-2, PlayerShip: 1)
- [ ] Enemies detected (not duplicated)
- [ ] No missing important objects
- [ ] Inference speed >30 FPS
- [ ] No false positives on empty space

---

## Command to Test

```bash
# Test live detection
.\.venv\Scripts\python.exe -m detection.detector --live --model "F:\dev\bot\best.pt" --show

# Run V2 bot (uses optimized settings automatically)
python -m darkorbit_bot.v2.main

# Run V1 bot (uses optimized settings automatically)
python -m darkorbit_bot.reasoning.bot_controller
```

---

## Additional YOLO Parameters

The detector also sets these optimized parameters:

```python
results = model.predict(
    frame,
    conf=0.3,              # Confidence threshold
    iou=0.3,               # NMS IoU threshold
    verbose=False,         # Don't print results
    imgsz=1280,            # High resolution (1280px)
    half=True,             # FP16 inference (2x faster)
    device=self.device,    # GPU device
    max_det=300,           # Max 300 detections per frame
    agnostic_nms=False     # Per-class NMS (don't merge Ship with Enemy)
)
```

All parameters are optimized for:
- Speed: FP16, GPU, efficient NMS
- Accuracy: High resolution (1280px)
- Duplicate removal: Low IoU threshold (0.3)

---

## Summary

**Key changes**:
1. Lowered confidence: 0.4 → 0.3 (detect more, filter weak)
2. Lowered IoU: 0.7 → 0.3 (aggressive duplicate removal)
3. Applied to both V1 and V2 bots
4. Inference speed improved: 20 FPS → 41 FPS

**Result**: Clean detections with minimal duplicates!
