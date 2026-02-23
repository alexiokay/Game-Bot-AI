# DarkOrbit Bot V2 - Complete Guide

## Quick Start

### Running the Bot

```bash
# Basic run (monitor 1, default policy)
python -m darkorbit_bot.v2.bot_controller_v2

# Run on specific monitor with custom policy
python -m darkorbit_bot.v2.bot_controller_v2 --monitor 2 --policy-dir v2/checkpoints

# With VLM corrections enabled
python -m darkorbit_bot.v2.bot_controller_v2 --policy-dir v2/checkpoints --vlm --vlm-corrections

# Shadow training (bot watches you play and trains executor)
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --shadow-lr 1e-4

# Shadow training + full recording (captures data for ALL 3 models)
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --save-recordings

# Online learning (bot learns from its own mistakes)
python -m darkorbit_bot.v2.bot_controller_v2 --online-learning --online-lr 1e-5
```

### Preview YOLO Detections

```bash
# Live preview of YOLO detections with bounding boxes
python -m detection.detector --live --model "F:\dev\bot\best.pt" --show

# Preview on specific monitor
python -m detection.detector --live --model best.pt --monitor 2 --show

# Save detection samples
python -m detection.detector --live --model best.pt --save-dir detections/
```

### Offline Training

```bash
# Train executor from recorded human demonstrations
python -m darkorbit_bot.v2.training.train_executor --data recordings/ --output-dir models/

# Train tactician (target selection)
python -m darkorbit_bot.v2.training.train_tactician --data recordings/ --output-dir models/

# Train strategist (mode selection)
python -m darkorbit_bot.v2.training.train_strategist --data recordings/ --output-dir models/

# Train full pipeline (all three models)
python -m darkorbit_bot.v2.training.train_full --data recordings/ --output-dir models/
```

### View Training Progress

```bash
# Start tensorboard (run in separate terminal)
tensorboard --logdir=runs

# Or with Python module
python -m tensorboard.main --logdir=runs

# Suppress warnings (optional)
python -W ignore -m tensorboard.main --logdir=runs

# Then open http://localhost:6006/ in browser
```

---

## Command Line Arguments

### Core Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--monitor` | `1` | Which monitor to capture (1, 2, etc.) |
| `--policy-dir` | `None` | Directory with policy files (strategist.pt, tactician.pt, executor.pt) |
| `--model-path` | `best.pt` | Path to YOLO detection model |
| `--device` | `cuda` | Device for inference (`cuda` or `cpu`) |

### Training Modes
| Argument | Default | Description |
|----------|---------|-------------|
| `--shadow-train` | `False` | Shadow training mode - bot watches you play and learns |
| `--shadow-lr` | `1e-4` | Learning rate for shadow training (10x higher than online) |
| `--online-learning` | `False` | Online learning - bot learns from hit/miss feedback |
| `--online-lr` | `1e-5` | Learning rate for online learning (conservative) |

### Visual Features
| Argument | Default | Description |
|----------|---------|-------------|
| `--visual-features` | `True` | Enable visual feature extraction |
| `--visual-lightweight` | `False` | Use lightweight color encoder (faster, no GPU) |
| `--hud-ocr` | `True` | Enable HUD OCR for HP/Shield reading |
| `--hud-backend` | `color_only` | OCR backend: `easyocr`, `tesseract`, `paddleocr`, `color_only` |

### VLM (Vision Language Model)
| Argument | Default | Description |
|----------|---------|-------------|
| `--vlm` | `False` | Enable VLM analysis |
| `--vlm-url` | `http://localhost:1234` | VLM server URL |
| `--vlm-model` | `local-model` | VLM model name |
| `--vlm-corrections` | `False` | Save VLM corrections for training |

### Auto-Labeling
| Argument | Default | Description |
|----------|---------|-------------|
| `--auto-label` | `False` | Enable auto-labeling during gameplay |
| `--auto-label-dir` | `data/auto_labeled` | Output directory for labeled data |
| `--auto-label-interval` | `600` | Sample every N frames |
| `--gemini-model` | `gemini-2.0-flash` | Gemini model for classification |

### Logging & Checkpointing
| Argument | Default | Description |
|----------|---------|-------------|
| `--log-dir` | `runs` | Directory for Tensorboard logs |
| `--checkpoint-keep-top` | `5` | Keep top N checkpoints by score |
| `--checkpoint-keep-latest` | `3` | Keep N most recent checkpoints |

---

## Hotkeys (During Runtime)

| Key | Action |
|-----|--------|
| `F1` | Start/Stop (pause/resume) |
| `F2` | BAD STOP - Save recent actions as negative training data |
| `F3` | Emergency stop |
| `F4` | Toggle debug logging |
| `F5` | Toggle reasoning log |
| `F6` | Cycle mode override (AUTO → FIGHT → LOOT → FLEE → EXPLORE → CAUTIOUS → AUTO) |

---

## Training Infrastructure

### 1. Tensorboard Logging

All training metrics are logged to Tensorboard for visualization.

```bash
# Install tensorboard
pip install tensorboard

# View training curves (run in separate terminal)
tensorboard --logdir=runs

# Or if that doesn't work:
python -m tensorboard.main --logdir=runs
```

Then open `http://localhost:6006` in your browser.

**Logged Metrics:**
- `Loss/total` - Combined training loss
- `Loss/position` - Mouse position prediction loss
- `Loss/click` - Click prediction loss
- `Metrics/position_error` - Average position error (0-1 range)
- `Metrics/click_accuracy` - Click prediction accuracy (0-100%)
- `Metrics/hit_rate` - Online learning hit rate
- `Metrics/buffer_size` - Experience buffer utilization
- `Gradients/norm` - Gradient magnitude (helps detect instability)
- `LR/learning_rate` - Current learning rate (shows warmup + decay)

### 2. Smart Checkpoint Management

Checkpoints are automatically managed with scoring and versioning.

**Features:**
- Keeps top N checkpoints by performance score
- Keeps N most recent checkpoints (regardless of score)
- Auto-cleanup of old checkpoints
- Metadata tracking (timestamp, metrics, config)
- Rollback capability

**Checkpoint Location:** `{policy_dir}/checkpoints/`

**Checkpoint Files:**
```
checkpoints/
  executor_v1_0.8234.pt    # version 1, score 0.8234
  executor_v2_0.8456.pt    # version 2, score 0.8456
  executor_v3_0.8123.pt    # version 3, score 0.8123
  checkpoint_index.json    # metadata for all checkpoints
```

**Programmatic Usage:**
```python
from darkorbit_bot.v2.training import CheckpointManager

manager = CheckpointManager("checkpoints/", model_name="executor", keep_top_n=5)

# Save checkpoint with score
manager.save(model, optimizer, score=0.85, step=100, metrics={"hit_rate": 0.9})

# Load best checkpoint
manager.load_best(model, optimizer)

# Load latest checkpoint
manager.load_latest(model, optimizer)

# Rollback to previous version
manager.rollback(model, optimizer, steps_back=1)

# List all checkpoints
for ckpt in manager.list_checkpoints():
    print(f"v{ckpt['version']}: score={ckpt['score']:.4f}")
```

### 3. Learning Rate Scheduling

Training uses warmup + cosine decay for stability.

**Schedule:**
1. **Warmup Phase** (first 50-100 steps): LR increases linearly from 0 to base_lr
2. **Decay Phase** (remaining steps): LR decays via cosine annealing to min_lr

**Why This Helps:**
- Warmup prevents early instability when gradients are noisy
- Cosine decay prevents overfitting late in training
- Smoother convergence than fixed LR

**Defaults:**
- Shadow Training: warmup=50 steps, total=5000 steps, min_lr=1% of base
- Online Learning: warmup=100 steps, total=10000 steps, min_lr=10% of base

### 4. Prioritized Experience Replay (Advanced)

For more efficient learning, use PrioritizedReplayBuffer instead of random sampling.

```python
from darkorbit_bot.v2.training import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(max_size=10000, alpha=0.6, beta=0.4)

# Add experience with priority
buffer.add(experience, priority=td_error)

# Sample batch (returns experiences, indices, importance weights)
experiences, indices, weights = buffer.sample(batch_size=32)

# Update priorities after computing new TD errors
buffer.update_priorities(indices, new_td_errors)
```

---

## Training Modes Explained

### Shadow Training (Behavioral Cloning)

**What it does:** Bot watches you play and learns to imitate your actions.

**How it works:**
1. You play the game normally
2. Bot captures your mouse position and clicks
3. Bot records (state, goal, target_info) → your_action pairs
4. Every few seconds, bot trains to minimize prediction error vs your actions

**Best for:**
- Initial training when bot doesn't know anything
- Teaching specific behaviors (click timing, target selection)
- Quick improvements (10x faster than online learning)

**Usage:**
```bash
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --shadow-lr 1e-4
```

**Tips:**
- Play consistently - bot learns whatever you do
- Play for at least 5-10 minutes to get meaningful data
- Higher LR (1e-4) is fine because human demonstrations are high quality

### Online Learning (Reinforcement Learning)

**What it does:** Bot learns from its own experience during gameplay.

**How it works:**
1. Bot plays the game
2. Bot detects hits (target HP dropped after click) and misses
3. Bot rewards itself for good actions, penalizes bad ones
4. Model weights update every few seconds

**Reward Signals:**
- **Hit:** +1.0 (clicked and target HP dropped)
- **Miss:** -0.3 (clicked but target HP unchanged)
- **Closer:** +0.2 (moved closer to target)
- **Farther:** -0.1 (moved away from target)

**Best for:**
- Fine-tuning after shadow training
- Adapting to specific enemy types
- Long-term improvement

**Usage:**
```bash
python -m darkorbit_bot.v2.bot_controller_v2 --online-learning --online-lr 1e-5
```

**Tips:**
- Use lower LR (1e-5) to avoid destabilizing
- Requires HUD OCR to detect HP changes
- Best combined with shadow training first

---

## Config Options (BotConfigV2)

```python
@dataclass
class BotConfigV2:
    # Detection
    model_path: str = "best.pt"
    monitor: int = 1

    # Policy
    policy_dir: Optional[str] = None
    device: str = "cuda"

    # Humanization
    reaction_delay_ms: int = 40
    precision_noise: float = 0.03

    # Safety
    max_actions_per_second: int = 60

    # Tracking
    tracker_high_thresh: float = 0.6
    tracker_low_thresh: float = 0.1
    tracker_buffer: int = 30

    # VLM
    vlm_enabled: bool = False
    vlm_url: str = "http://localhost:1234"
    vlm_model: str = "local-model"
    vlm_corrections: bool = False

    # Online Learning
    online_learning: bool = False
    online_learning_rate: float = 1e-5
    online_update_interval: float = 5.0

    # Auto-Labeling
    auto_label: bool = False
    auto_label_dir: str = "data/auto_labeled"
    auto_label_interval: int = 600
    gemini_api_key: str = None
    gemini_model: str = "gemini-2.0-flash"

    # Visual Features
    visual_features: bool = True
    visual_lightweight: bool = False

    # HUD OCR
    hud_ocr_enabled: bool = True
    hud_ocr_backend: str = "color_only"

    # Shadow Training
    shadow_train: bool = False
    shadow_train_lr: float = 1e-4

    # Smart Checkpointing
    checkpoint_keep_top_n: int = 5
    checkpoint_keep_latest_n: int = 3
    checkpoint_save_interval: int = 100

    # Tensorboard Logging
    log_dir: str = "runs"
```

---

## Architecture Overview

### Hierarchical Policy (3-Level)

```
STRATEGIST (1 Hz)     → Goal selection (FIGHT, LOOT, FLEE, EXPLORE, CAUTIOUS)
     ↓
TACTICIAN (10 Hz)     → Target selection (which enemy/loot to focus)
     ↓
EXECUTOR (60 Hz)      → Motor control (mouse position, click timing)
```

### Data Flow

```
Screen Capture → YOLO Detection → ByteTrack → State Encoder → Policy → Humanizer → Mouse/Keyboard
                                     ↓
                              Visual Encoder (optional)
                                     ↓
                              HUD OCR (HP/Shield)
```

---

## File Structure

```
darkorbit_bot/v2/
├── bot_controller_v2.py    # Main controller
├── config.py               # Configuration classes
├── GUIDE.md                # This guide
│
├── models/                 # Neural network architectures
│   ├── unified.py          # HierarchicalPolicy (Strategist + Tactician + Executor)
│   ├── executor.py         # Executor (Mamba/LSTM motor control)
│   ├── tactician.py        # Tactician (target selection)
│   └── strategist.py       # Strategist (mode selection)
│
├── perception/             # Perception pipeline
│   ├── tracker.py          # ByteTrack object tracking
│   ├── state_encoder.py    # State encoding for policy
│   ├── vision_encoder.py   # Visual feature extraction
│   ├── hud_ocr.py          # HP/Shield reading from HUD
│   └── visual_outcome_detector.py  # AI-based event detection
│
├── training/               # Training infrastructure
│   ├── training_utils.py   # Tensorboard, checkpoints, LR scheduling
│   ├── shadow_trainer.py   # Behavioral cloning from human play
│   ├── online_learner.py   # Online RL from hit/miss feedback
│   ├── train_executor.py   # Offline executor training
│   ├── train_tactician.py  # Offline tactician training
│   ├── train_strategist.py # Offline strategist training
│   └── train_full.py       # Full pipeline training
│
└── vlm/                    # Vision Language Model integration
    └── vlm_v2.py           # VLM analysis and corrections
```

---

## Complete Training Workflows

### Workflow 1: Training from Scratch (Recommended)

```bash
# Step 1: Preview YOLO detections to verify detection model works
python -m detection.detector --live --model best.pt --show

# Step 2: Run shadow training for 15-20 minutes
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --shadow-lr 1e-4 --policy-dir v2/models

# Step 3: In another terminal, monitor training
tensorboard --logdir=runs

# Step 4: After shadow training, run online learning to fine-tune
python -m darkorbit_bot.v2.bot_controller_v2 --online-learning --online-lr 1e-5 --policy-dir v2/models

# Step 5: Check best checkpoint
# Checkpoints saved to: v2/models/checkpoints/executor_v*_*.pt
```

**What to monitor in Tensorboard:**
- Shadow training: `position_error` should drop below 0.1, `click_accuracy` above 70%
- Online learning: `hit_rate` should increase over time (target: >60%)

### Workflow 2: Fine-tuning Existing Model

```bash
# Step 1: Load existing model and run online learning
python -m darkorbit_bot.v2.bot_controller_v2 \
    --policy-dir v2/models \
    --online-learning \
    --online-lr 5e-6

# Step 2: Monitor for improvements
tensorboard --logdir=runs

# Step 3: Compare checkpoints
# Best checkpoints kept in v2/models/checkpoints/
```

### Workflow 3: VLM-Guided Improvements

```bash
# Step 1: Enable VLM corrections during gameplay
python -m darkorbit_bot.v2.bot_controller_v2 \
    --policy-dir v2/models \
    --vlm \
    --vlm-corrections \
    --online-learning

# VLM will analyze mistakes and save corrections
# Corrections saved to: vlm_corrections/

# Step 2: Review VLM feedback in logs
# VLM provides strategic advice on mode selection, targeting, etc.
```

### Workflow 4: Full Hierarchical Recording for Offline Training

**NEW FEATURE**: Shadow training can now capture **full hierarchical data** for training all 3 models!

```bash
# Step 1: Run shadow training with full recording enabled
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --save-recordings

# This captures:
# - Executor data: Your mouse movements and clicks
# - Tactician data: Which enemies you target (inferred from mouse position)
# - Strategist data: Mode decisions (FIGHT/LOOT/FLEE inferred from behavior)

# Step 2: Play for 20-30 minutes (more data = better offline training)
# Recordings automatically saved to data/recordings/ directory

# Step 3: Train all models offline from your recordings
python -m darkorbit_bot.v2.training.train_full \
    --data data/recordings \
    --output-dir v2/models \
    --epochs 50 \
    --batch-size 64

# Step 4: Test trained model
python -m darkorbit_bot.v2.bot_controller_v2 --policy-dir v2/models
```

**How Target Selection is Inferred:**
- Executor: Direct mouse position/click capture (100% accurate)
- Tactician: Inferred from which enemy is closest to your mouse (within 0.15 screen distance)
- Strategist: Inferred from combat patterns (currently uses bot's mode as baseline)

**Benefits over Standard Shadow Training:**
- Standard `--shadow-train`: Only trains executor
- With `--save-recordings`: Captures data for **all 3 models** + enables offline training

---

## Best Practices for V2

### 1. YOLO Detection Model

**Quality Check:**
```bash
# Always preview detections before training
python -m detection.detector --live --model best.pt --show --monitor 2
```

**What to look for:**
- ✅ Enemies detected with confidence > 0.6
- ✅ BonusBoxes detected reliably
- ✅ No false positives on background
- ❌ If detections are poor, retrain YOLO first

**Improving YOLO:**
- Use auto-labeling: `--auto-label --gemini-api-key YOUR_KEY`
- Collect diverse data (different maps, ships, enemies)
- Minimum 500 images per class

**Understanding YOLO vs Visual Features:**

YOLO provides the **foundation** that models always use:
- Bounding boxes: (x, y, width, height) for each detected object
- Class labels: Enemy, BonusBox, etc.
- Confidence scores
- Velocity tracking (from ByteTrack)

Visual features are **optional enhancements**:
- CNN-based pixel analysis of each object's bounding box
- Captures: health bars, shields, boss indicators, status effects
- More compute intensive but provides richer context
- Disabled by default, enable with `--visual-features` (or disable with `--no-visual`)

**The models ALWAYS learn from YOLO detections** - visual features just add extra context on top!

### 2. Shadow Training Best Practices

**Do's:**
✅ Play consistently - bot learns your exact behavior
✅ Focus on one skill at a time (aiming, tracking, mode switching)
✅ Play for 15-20 minutes minimum
✅ Check Tensorboard: `position_error < 0.1`, `click_accuracy > 70%`
✅ Use higher LR (1e-4) - human demonstrations are high quality

**Don'ts:**
❌ Don't play erratically or randomly
❌ Don't AFK - bot learns idle behavior
❌ Don't switch between different play styles mid-session
❌ Don't use LR > 1e-3 (causes instability)

**Monitoring Shadow Training:**
```bash
# Watch these metrics in Tensorboard:
Loss/position      # Should decrease to < 0.5
Metrics/position_error  # Should be < 0.1 (10% of screen)
Metrics/click_accuracy  # Should be > 70%
LR/learning_rate   # Should warm up then decay
```

**Full Recording Mode (--save-recordings):**

Use `--save-recordings` with shadow training to capture data for ALL 3 models:

```bash
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --save-recordings
```

**What gets captured:**
- ✅ Executor: Your exact mouse movements and clicks
- ✅ Tactician: Which enemy you're targeting (inferred from mouse position)
- ✅ Strategist: Mode decisions (inferred from combat behavior)
- ✅ Frame data: Screenshots with YOLO detections
- ✅ Tracked objects: All enemies/loot with IDs and velocities

**Recordings saved to:** `data/recordings/shadow_recording_YYYYMMDD_HHMMSS.pkl`

**Use recordings for offline training:**
```bash
# Train all 3 models from your gameplay recordings
python -m darkorbit_bot.v2.training.train_full --data data/recordings --output-dir v2/models
```

**Key difference:**
- Standard `--shadow-train`: Only trains executor in real-time
- With `--save-recordings`: Trains executor + saves full data for offline training of all 3 models

### 3. Online Learning Best Practices

**Do's:**
✅ Start with shadow-trained model first
✅ Use very low LR (1e-5 or 1e-6)
✅ Enable HUD OCR for accurate hit detection
✅ Run for extended sessions (30+ minutes)
✅ Monitor `hit_rate` - should improve over time

**Don'ts:**
❌ Don't use online learning from random initialization
❌ Don't use high LR (causes forgetting)
❌ Don't disable HUD OCR (no reward signal)
❌ Don't expect fast results (RL is slower than BC)

**Monitoring Online Learning:**
```bash
# Watch these metrics in Tensorboard:
Metrics/hit_rate        # Should increase over time
Metrics/avg_reward      # Should be positive on average
Loss/total              # May fluctuate (normal for RL)
Gradients/norm          # Should be < 10 (stable)
```

### 4. Checkpoint Management

**Understanding Scores:**
```python
# Checkpoints ranked by combined score:
score = 0.4 * hit_rate + 0.2 * click_accuracy - 0.2 * position_error
```

**Best Practices:**
- Keep top 5 checkpoints (default)
- Test best checkpoint before production use
- Rollback if new checkpoint performs worse
- Archive good checkpoints periodically

**Manual Checkpoint Management:**
```python
from darkorbit_bot.v2.training import CheckpointManager

manager = CheckpointManager("v2/models/checkpoints", model_name="executor")

# List all checkpoints by score
for ckpt in manager.list_checkpoints():
    print(f"v{ckpt['version']}: score={ckpt['score']:.4f} @ {ckpt['timestamp']}")

# Load specific version
manager.load_version(version=5, model=executor, optimizer=optimizer)

# Rollback to previous
manager.rollback(model=executor, optimizer=optimizer, steps_back=1)
```

### 5. Learning Rate Guidelines

| Training Type | Recommended LR | Warmup Steps | Notes |
|---------------|----------------|--------------|-------|
| Shadow Training | 1e-4 | 50 | High quality data, fast convergence |
| Online Learning (new) | 1e-5 | 100 | Conservative, stable |
| Online Learning (fine-tune) | 5e-6 | 50 | Very conservative |
| Offline Executor Training | 3e-4 | 100 | Batch training, can be higher |
| VLM-Guided Training | 1e-5 | 100 | Trusted corrections, moderate |

**Rule of thumb:**
- Human demonstrations → higher LR (1e-4)
- Self-play feedback → lower LR (1e-5)
- Fine-tuning → even lower LR (5e-6)

### 6. Debugging Poor Performance

**Bot not clicking:**
```bash
# 1. Check HUD OCR is working
# Look for HP values in console output

# 2. Check click predictions in Tensorboard
# Metrics/click_accuracy should be > 50%

# 3. Run shadow training focused on clicking
# Play aggressively with lots of clicks

# 4. Check executor output
# action[2] > 0 means click, < 0 means no click
```

**Bot moving to wrong locations:**
```bash
# 1. Preview YOLO detections
python -m detection.detector --live --model best.pt --show

# 2. Verify screen capture resolution
# Check bot logs for screen dimensions

# 3. Run shadow training
# Record correct mouse movements

# 4. Check position error in Tensorboard
# Should be < 0.1 (10% of screen)
```

**Bot making poor decisions:**
```bash
# 1. Enable VLM corrections
python -m darkorbit_bot.v2.bot_controller_v2 --vlm --vlm-corrections

# 2. Check strategist/tactician weights
# May need offline training for strategic decisions

# 3. Review mode override (F6 key)
# Manually set correct mode to see if executor works

# 4. Check tracked objects
# F4 for debug logs - verify correct objects detected
```

### 7. Production Deployment

**Recommended Setup:**
```bash
# 1. Train with shadow + online learning
# 2. Select best checkpoint from v2/models/checkpoints/
# 3. Copy to production directory
cp v2/models/checkpoints/executor_v10_0.8567.pt v2/production/executor.pt

# 4. Run in production (no training)
python -m darkorbit_bot.v2.bot_controller_v2 \
    --policy-dir v2/production \
    --monitor 2

# 5. Monitor performance metrics
# Log K/D ratio, loot collected, etc.
```

### 8. Continuous Improvement Loop

```
Week 1: Shadow training (15-20 min/day) → Checkpoint v1-v5
Week 2: Online learning (1 hour/day) → Checkpoint v6-v10
Week 3: Evaluate best checkpoint → Select v8 (score 0.856)
Week 4: Production use + collect edge cases
Week 5: Shadow training on edge cases → Checkpoint v11-v15
Week 6: Fine-tune with online learning (LR 5e-6) → Checkpoint v16-v20
...repeat...
```

**Metrics to Track:**
- Hit rate (target: >70%)
- K/D ratio (target: >2.0)
- Loot efficiency (loot/time)
- Death rate (minimize)
- Mode selection accuracy (via VLM analysis)

---

## Troubleshooting

### Tensorboard not found
```bash
pip install tensorboard
# Or use:
python -m tensorboard.main --logdir=runs
```

### CUDA out of memory
```bash
# Use CPU instead
python -m darkorbit_bot.v2.bot_controller_v2 --device cpu

# Or use lightweight visual encoder
python -m darkorbit_bot.v2.bot_controller_v2 --visual-lightweight
```

### Bot not clicking
- Check HUD OCR is detecting targets correctly
- Try shadow training to teach click timing
- Lower the click threshold by editing executor behavior

### Bot moving to wrong position
- Ensure correct monitor is selected (`--monitor 2`)
- Check that screen coordinates are being captured correctly
- Run shadow training to recalibrate

### Training loss not decreasing
- Check gradient norms in Tensorboard (should be < 10)
- Try lower learning rate
- Ensure you have enough training data (min 100 samples)

### Checkpoints not saving
- Ensure `policy_dir` is set
- Check disk space
- Verify write permissions to checkpoint directory

---

## Best Practices

### For Shadow Training
1. Play consistently and intentionally
2. Focus on the behavior you want to teach
3. Train for 10-20 minutes minimum
4. Check Tensorboard for position_error < 0.1

### For Online Learning
1. Start with a shadow-trained model
2. Use very low LR (1e-5 or lower)
3. Monitor hit_rate in Tensorboard
4. Let it run for extended sessions (30+ min)

### For Production Use
1. Disable training modes
2. Use best checkpoint from training
3. Monitor performance metrics
4. Periodically retrain on new data

---

## Quick Reference: Common Commands

### Daily Use

```bash
# Run bot (production)
python -m darkorbit_bot.v2.bot_controller_v2 --policy-dir v2/models --monitor 2

# View training progress
tensorboard --logdir=runs

# Preview YOLO detections
python -m detection.detector --live --model best.pt --show
```

### Training Sessions

```bash
# Shadow training (15-20 min)
python -m darkorbit_bot.v2.bot_controller_v2 --shadow-train --shadow-lr 1e-4 --policy-dir v2/models

# Online learning (30+ min)
python -m darkorbit_bot.v2.bot_controller_v2 --online-learning --online-lr 1e-5 --policy-dir v2/models

# Fine-tuning (existing model)
python -m darkorbit_bot.v2.bot_controller_v2 --online-learning --online-lr 5e-6 --policy-dir v2/models
```

### Advanced

```bash
# VLM-guided training
python -m darkorbit_bot.v2.bot_controller_v2 --policy-dir v2/models --vlm --vlm-corrections --online-learning

# Auto-labeling for YOLO improvement
python -m darkorbit_bot.v2.bot_controller_v2 --auto-label --gemini-api-key YOUR_KEY

# Offline training from recordings
python -m darkorbit_bot.v2.training.train_full --data recordings/ --output-dir v2/models
```

### Checkpoint Management (Python)

```python
from darkorbit_bot.v2.training import CheckpointManager

manager = CheckpointManager("v2/models/checkpoints", model_name="executor")
manager.list_checkpoints()  # View all
manager.load_best(model, optimizer)  # Load best
manager.rollback(model, optimizer, steps_back=1)  # Undo last
```

---

## Version History

- **V2.0**: Initial hierarchical architecture
- **V2.1**: Added shadow training, online learning
- **V2.2**: Added Tensorboard, smart checkpointing, LR scheduling

---

## Credits & References

**Architecture:**
- Hierarchical Policy: Strategist → Tactician → Executor (1Hz → 10Hz → 60Hz)
- ByteTrack: Multi-object tracking with temporal consistency
- Mamba/LSTM: Sequential decision making for executor

**Training Infrastructure:**
- Tensorboard: Real-time training visualization
- Behavioral Cloning: Direct imitation from human demonstrations
- Online RL: Self-supervised learning from hit/miss feedback
- Weak Supervision: Auto-labeling from game state signals

**Tools:**
- YOLO: Real-time object detection
- PyTorch: Deep learning framework
- LM Studio: Local VLM server (optional)
- Gemini API: Auto-labeling for YOLO (optional)
