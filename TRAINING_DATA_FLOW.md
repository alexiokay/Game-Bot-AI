# Training Data Flow - Complete Guide

This document explains how ALL data is used efficiently in the DarkOrbit bot training pipeline.

## Data Sources

### 1. **V2 Recorder** (Pure Recording)
- **Location**: `darkorbit_bot/data/recordings_v2/sequence_*.json`
- **Format**: JSON with states, actions, objects
- **Created by**: Pure Recording button (no models needed)
- **Sample extraction**: Sliding window (stride = 25% of history length)
- **Example**: 900 frames → ~57 training samples

### 2. **Shadow Training** (Imitation Learning)
- **Location**: `darkorbit_bot/data/recordings_v2/shadow_recording_*.json`
- **Format**: JSON with demos (one per frame)
- **Created by**: Shadow Training button (needs Executor model)
- **Sample extraction**: 1 sample per demo
- **Example**: 54,000 frames → 54,000 training samples
- **Real-time training**: Executor trained every 3 seconds during play

### 3. **VLM Corrections** (Expert Feedback)
- **Location**: `darkorbit_bot/data/vlm_corrections_v2/v2_corrections_*.json`
- **Format**: JSON with VLM critiques and suggested corrections
- **Created by**: Running bot with `--vlm --vlm-corrections`
- **Used by**: Fine-tuning (not initial training)
- **Contains**: Corrections for Strategist, Tactician, and Executor

---

## Training Pipeline

### Phase 1: Bootstrap (No Models Yet)

```mermaid
V2 Recorder → recordings_v2/*.json → Train All 3 Models
```

**What happens**:
1. Use **Pure Recording** to record yourself playing (no models needed)
2. Saves to `recordings_v2/sequence_*.json`
3. Train all 3 models from scratch:
   - `Train Strategist` → learns mode selection (FIGHT/LOOT/FLEE/EXPLORE)
   - `Train Tactician` → learns target selection
   - `Train Executor` → learns mouse movement

**Data efficiency**:
- Sliding window extracts maximum samples from sequences
- 1 hour of recording → ~3,400 samples (at 30 FPS, 16-frame history)

### Phase 2: Shadow Training (Models Exist)

```mermaid
Shadow Training → Real-time Executor updates + recordings_v2/*.json
                → Re-train Strategist/Tactician offline
```

**What happens**:
1. Use **Shadow Training** with `--save-recordings`
2. **Executor** trains in real-time (every 3 seconds)
3. **Strategist/Tactician** data saved for offline re-training
4. Run `Train Strategist` and `Train Tactician` again with ALL data (Phase 1 + Phase 2)

**Data efficiency**:
- 1 sample per frame (much denser than V2 Recorder)
- Executor gets immediate feedback
- Strategist/Tactician benefit from more diverse situations

### Phase 3: VLM Fine-tuning (Optional Refinement)

```mermaid
Run Bot with --vlm --vlm-corrections → vlm_corrections_v2/*.json
                                     → Fine-tune with VLM button
```

**What happens**:
1. Run bot autonomously with VLM enabled
2. VLM critiques bot's decisions and saves corrections
3. Use **Fine-tune with VLM** button to improve models with expert feedback

**Data efficiency**:
- Only corrects mistakes (not redundant with good plays)
- Uses VLM's superior understanding to fix edge cases
- Lower learning rate (5e-5 vs 1e-4) for careful refinement

---

## Complete Workflow

### Recommended Training Path

1. **Bootstrap** (1 hour recording + 4-6 hours training)
   ```bash
   # Record yourself playing
   Pure Recording → 3,600 frames → ~225 samples

   # Train all models
   Train Strategist (100 epochs)
   Train Tactician (80 epochs)
   Train Executor (60 epochs)
   ```

2. **Shadow Training** (Play + Learn)
   ```bash
   # Play while Executor learns in real-time
   Shadow Training (with --save-recordings)
   → Executor improves every 3 seconds
   → Strategist/Tactician data accumulated

   # Re-train upper layers with combined data
   Train Strategist (Phase 1 + Phase 2 data)
   Train Tactician (Phase 1 + Phase 2 data)
   ```

3. **VLM Refinement** (Optional)
   ```bash
   # Let bot play with VLM watching
   Run Bot (with --vlm --vlm-corrections) → 30 min

   # Fine-tune with corrections
   Fine-tune Executor with VLM (20 epochs)
   Fine-tune Strategist with VLM (20 epochs)
   Fine-tune Tactician with VLM (20 epochs)
   ```

---

## Data Usage by Training Script

### train_strategist.py
**Loads**:
- ✅ V2 Recorder JSON (`sequence_*.json`)
- ✅ Shadow Training JSON (`shadow_recording_*.json`)
- ✅ Old NPZ/PKL files (legacy support)

**Extracts**:
- V2 Recorder: Sliding window samples
- Shadow Training: 1 sample per demo (repeated state for history)

**Does NOT use**:
- ❌ VLM corrections (use `finetune_with_vlm.py` instead)

### train_tactician.py
**Loads**:
- ✅ V2 Recorder JSON
- ✅ Shadow Training JSON
- ✅ Old NPZ/PKL files

**Extracts**:
- Similar to Strategist
- Focuses on target selection patterns

**Does NOT use**:
- ❌ VLM corrections

### train_executor.py
**Loads**:
- ✅ V2 Recorder JSON
- ✅ Shadow Training JSON
- ✅ Old NPZ/PKL files

**Extracts**:
- Frame-by-frame action predictions
- Object features for context

**Does NOT use**:
- ❌ VLM corrections

### finetune_with_vlm.py
**Loads**:
- ✅ VLM corrections ONLY (`vlm_corrections_v2/*.json`)

**Uses**:
- VLM's suggested actions as ground truth
- Fine-tunes pre-trained models (not train from scratch)

**Requires**:
- Pre-trained models from Phase 1 or 2

---

## Data Efficiency Summary

| Data Source | Samples/Hour | Training Phase | Real-time | Models Trained |
|-------------|--------------|----------------|-----------|----------------|
| V2 Recorder | ~225 | Bootstrap | ❌ No | All 3 |
| Shadow Training | ~108,000 | Improvement | ✅ Executor only | All 3 (offline) |
| VLM Corrections | Varies | Refinement | ❌ No | All 3 (fine-tune) |

**Key Insights**:
1. **Shadow Training is 480x denser** than V2 Recorder (1 sample/frame vs sliding window)
2. **All data is used** - nothing wasted
3. **VLM corrections are additive** - they complement, not replace, other data
4. **Training scripts automatically merge** all compatible formats

---

## Launcher Features

### Buttons Added
- ✅ **Shadow Training** - Real-time Executor training + recording
- ✅ **Pure Recording** - Bootstrap without models
- ✅ **Train Strategist** - Uses all recordings_v2 data
- ✅ **Train Tactician** - Uses all recordings_v2 data
- ✅ **Train Executor** - Uses all recordings_v2 data
- ✅ **Fine-tune with VLM** - Uses VLM corrections (NEW!)
- ✅ **Data Statistics** - Shows what data you have
- ✅ **Evaluate Models** - Checks model performance
- ✅ **View TensorBoard** - Visualize training metrics

### Configuration (Right-click any button)
- All parameters are configurable
- Settings persist in `launcher_config.json`
- Defaults are optimized for best results

---

## FAQ

**Q: Do I need to delete old recordings before Shadow Training?**
A: No! Training scripts merge all data. More data = better models.

**Q: Should I use VLM corrections from the start?**
A: No. Train basic models first (Phase 1), then optionally use VLM for refinement.

**Q: How often should I re-train?**
A: After collecting significant new data:
- After 1 hour of Shadow Training
- After bot plays 30+ min with VLM corrections
- When you notice bot making systematic mistakes

**Q: Can I train on multiple machines?**
A: Yes! Copy `recordings_v2/` folder to another machine and train there. Models are portable.

**Q: What if I only have 10 minutes of recording?**
A: That's ~600 frames → ~38 samples with V2 Recorder. Minimum for basic training, but you'll need more for good performance. Aim for 30-60 minutes minimum.

---

## Next Steps

1. **Test the launcher** - Try the new Fine-tune with VLM button
2. **Check data statistics** - Click "Data Statistics" to see what you have
3. **Start with Pure Recording** - Get 1 hour of good gameplay
4. **Train baseline models** - Run all 3 training buttons
5. **Enable Shadow Training** - Let Executor improve while you play
6. **Optional VLM** - Fine-tune after baseline is working
