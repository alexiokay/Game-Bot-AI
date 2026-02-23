# DarkOrbit Bot Commands Reference

Complete reference for all CLI commands, arguments, and hotkeys.

---

## V1 Commands (Current)

### Bot Controller

The main bot that plays the game using your trained policy.

```bash
python reasoning/bot_controller.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | path | required | Path to YOLO model (.pt) |
| `--policy` | path | required | Path to policy network (.pt) |
| `--monitor` | int | 0 | Monitor index (0 = primary) |
| `--noise` | float | 0.02 | Precision noise (0-0.1) for humanization |
| `--self-improve` | flag | off | Enable basic VLM critique (requires LM Studio) |
| `--enhanced-vlm` | flag | off | Enable multi-level VLM analysis (Strategic/Tactical/Execution) |
| `--auto-mode` | flag | off | Let network decide PASSIVE/AGGRESSIVE (learned from gameplay) |
| `--meta-learn` | flag | off | Auto-run meta-analysis at session end |
| `--no-prompt` | flag | off | Skip all interactive prompts (for automation) |

**Examples:**
```bash
# Basic bot run
python reasoning/bot_controller.py --model models/best.pt --policy models/policy.pt

# With VLM watching and corrections
python reasoning/bot_controller.py --policy data/checkpoints/policy_latest.pt --enhanced-vlm

# Full featured with meta-learning
python reasoning/bot_controller.py --policy data/checkpoints/policy_latest.pt --enhanced-vlm --meta-learn

# Automated (no prompts)
python reasoning/bot_controller.py --policy data/checkpoints/policy_latest.pt --enhanced-vlm --meta-learn --no-prompt
```

**Hotkeys (during bot run):**

| Key | Action |
|-----|--------|
| `F1` | Pause/Resume bot |
| `F2` | BAD STOP - save last ~2s as negative training data |
| `F3` | Emergency stop (kill switch) |
| `F4` | Toggle debug logging |
| `F5` | Toggle reasoning log |
| `F6` | Toggle mode override (PASSIVE/AGGRESSIVE/AUTO) |

---

### Filtered Recorder

Record your gameplay to train the bot.

```bash
python reasoning/filtered_recorder.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | path | required | Path to YOLO model for detection |
| `--monitor` | int | 0 | Monitor index |
| `--buffer` | float | 5.0 | Buffer seconds before kill/pickup |
| `--save-before` | float | 3.0 | Seconds to save before event |
| `--fps` | int | 20 | Target FPS for recording |

**Hotkeys (during recording):**

| Key | Action |
|-----|--------|
| `K` | Save current buffer as good gameplay |
| `L` | Mark kill event (auto-save) |
| `P` | Mark pickup event (auto-save) |
| `Esc` | Stop recording |

**Output:** `data/recordings/recording_TIMESTAMP.json`

---

### Training

Train the policy network from recordings.

```bash
python train.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--corrections` | flag | off | Include VLM corrections in training |
| `--corrections-only` | flag | off | Train only on corrections (skip recordings) |
| `--finetune` | flag | off | Fine-tune existing model (preserves weights) |

**Examples:**
```bash
# Initial training from recordings
python train.py --epochs 10

# Include VLM corrections
python train.py --corrections --epochs 15

# Quick fine-tune with just corrections
python train.py --finetune --corrections-only

# Full retrain with everything
python train.py --corrections --epochs 20
```

**Output:** `data/checkpoints/policy_latest.pt`, `data/checkpoints/policy_best.pt`

---

### VLM Meta-Learner

Analyze VLM performance and improve the system prompt.

```bash
python reasoning/vlm_meta_learner.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hours` | int | 24 | How many hours back to analyze |
| `--apply` | flag | off | Apply existing suggestions (no new analysis) |
| `--analyze` | flag | off | Run new analysis |
| `--url` | str | http://localhost:1234 | LM Studio API URL |

**Examples:**
```bash
# Run new analysis
python reasoning/vlm_meta_learner.py

# Apply previous suggestions (shows diff, asks confirmation)
python reasoning/vlm_meta_learner.py --apply

# Analyze last 2 hours only
python reasoning/vlm_meta_learner.py --hours 2

# Run new analysis AND apply
python reasoning/vlm_meta_learner.py --analyze --apply
```

**Output:**
- `data/meta_analysis/meta_analysis_TIMESTAMP.json` - Raw analysis
- `data/vlm_prompt_suggestions.md` - Human-readable suggestions
- `data/vlm_system_prompt.txt.backup` - Backup before changes

---

### Movement Pattern Analyzer

Analyze your mouse/keyboard patterns for humanization.

```bash
python analysis/analyze_patterns.py
```

No arguments - loads all recordings from `data/recordings/`.

**Output:** `data/my_movement_profile.json`

---

## File Locations

| What | Location |
|------|----------|
| **Models** | |
| YOLO model | `models/best.pt` |
| Policy network | `models/policy.pt` or `data/checkpoints/policy_latest.pt` |
| **Recordings** | |
| Gameplay recordings | `data/recordings/*.json` |
| Movement profile | `data/my_movement_profile.json` |
| **VLM Corrections** | |
| Self-improve corrections | `data/vlm_corrections/session_*/corrections.json` |
| Enhanced VLM corrections | `data/vlm_corrections/enhanced_*.json` |
| Bad stop corrections | `data/vlm_corrections/bad_stop_*.json` |
| Debug screenshots | `data/vlm_corrections/session_*/debug/` |
| **VLM Config** | |
| System prompt | `data/vlm_system_prompt.txt` |
| Prompt suggestions | `data/vlm_prompt_suggestions.md` |
| Meta-analysis | `data/meta_analysis/meta_analysis_*.json` |

---

## V2 Commands (Planned)

*These commands are planned for future versions.*

### Reinforcement Learning

```bash
# Planned: Train with RL instead of imitation learning
python train_rl.py --env darkorbit --policy models/policy.pt
```

### Multi-Agent

```bash
# Planned: Run multiple bot instances
python multi_agent.py --instances 3 --policy models/policy.pt
```

### Auto-Tuner

```bash
# Planned: Automatically tune hyperparameters
python auto_tune.py --target accuracy --trials 100
```

---

## Quick Start Workflow

### 1. Record Gameplay
```bash
python reasoning/filtered_recorder.py --model models/best.pt
# Play the game, press K to save good sequences
```

### 2. Analyze Patterns
```bash
python analysis/analyze_patterns.py
```

### 3. Train Model
```bash
python train.py --epochs 10
```

### 4. Run Bot
```bash
python reasoning/bot_controller.py --policy data/checkpoints/policy_latest.pt --enhanced-vlm
# Press F1 to start, F2 for bad behavior, F3 to stop
```

### 5. Improve with VLM
```bash
# After session, apply meta-learning suggestions
python reasoning/vlm_meta_learner.py --apply
```

### 6. Retrain with Corrections
```bash
python train.py --corrections --finetune
```

---

## Environment Requirements

- Python 3.10+
- CUDA (recommended for YOLO and training)
- LM Studio running at `localhost:1234` (for VLM features)
- Vision model loaded in LM Studio (e.g., `qwen/qwen3-vl-8b`)

---

## Troubleshooting

### Bot not attacking
- Check if Ctrl key is being pressed (F5 to see reasoning log)
- Verify YOLO is detecting enemies (red boxes visible)
- Try increasing `--noise` for more varied movement

### VLM not connecting
- Verify LM Studio is running at http://localhost:1234
- Check a vision model is loaded
- Test with: `curl http://localhost:1234/v1/models`

### Training loss not decreasing
- Need more recordings (aim for 10+ minutes)
- Check recordings have kills/pickups (event-based saving)
- Try lowering learning rate: `--lr 0.0005`

### Meta-learner no changes applied
- Check `data/vlm_prompt_suggestions.md` for suggestions
- The LLM may suggest changes that don't match existing text
- Manually edit `data/vlm_system_prompt.txt` if needed
