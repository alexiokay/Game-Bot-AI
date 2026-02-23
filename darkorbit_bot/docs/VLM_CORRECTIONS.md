# VLM Corrections System

This document explains how the VLM (Vision Language Model) correction systems work and how they influence model training.

---

## Overview

The bot has multiple systems that generate training corrections:

| System | Flag/Key | Analysis Frequency | Purpose |
|--------|----------|-------------------|---------|
| **Self-Improve** | `--self-improve` | Every ~3 seconds | Basic VLM critique, generates corrections |
| **Enhanced VLM** | `--enhanced-vlm` | Multi-level (0.3s-5s) | Richer VLM analysis, generates corrections |
| **Bad Stop** | `F2` key | Manual | YOU mark bot behavior as wrong |

VLM systems **observe only** - they don't control the bot at runtime. They generate training data that improves the model when you retrain.

**Bad Stop (F2)** is the simplest and most direct: when the bot does something wrong, press F2 to save the last ~2 seconds as "what NOT to do". No VLM needed!

---

## How It Works

### The Correction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     RUNTIME (Bot Playing)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Bot takes action based on policy network                    │
│           │                                                     │
│           ▼                                                     │
│  2. VLM sees: screenshot + bot's action + game state            │
│           │                                                     │
│           ▼                                                     │
│  3. VLM critiques: "Action was WRONG, should have done X"       │
│           │                                                     │
│           ▼                                                     │
│  4. Saves correction to disk:                                   │
│     - state_vector (what bot saw - 128/134 features)            │
│     - bot_action (what bot did - WRONG)                         │
│     - vlm_correction (what it SHOULD have done)                 │
│     - weight (how bad the mistake was: 1.0-2.5)                 │
│                                                                 │
│  Corrections saved to: data/vlm_corrections/                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING (Later)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  python train.py --corrections                                  │
│           │                                                     │
│           ▼                                                     │
│  Loads:                                                         │
│    - Your recordings (good gameplay examples)                   │
│    - VLM corrections (mistakes to avoid)                        │
│           │                                                     │
│           ▼                                                     │
│  Training:                                                      │
│    - Recordings teach: "This is how to play"                    │
│    - Corrections teach: "Don't do THIS, do THAT instead"        │
│    - Bad mistakes have higher weight (learned more strongly)    │
│           │                                                     │
│           ▼                                                     │
│  Result: Model that avoids past mistakes                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Self-Improve (`--self-improve`)

### How It Works

1. **Captures frames** every ~3 seconds during bot gameplay
2. **Sends to VLM** with screenshot + bounding boxes + bot state
3. **VLM responds** with:
   - Was action good/bad/needs_improvement?
   - What should the correct action be?
   - What tactic was being used?
4. **Saves corrections** to `data/vlm_corrections/session_*/corrections.json`

### Correction Format

```json
{
  "corrections": [
    {
      "timestamp": 1234567890.123,
      "quality": "bad",
      "state_vector": [0.1, 0.2, ...],  // 128 or 134 floats
      "bot_action": {
        "move_x": 0.5,
        "move_y": 0.3,
        "clicked": false
      },
      "correct_action": {
        "move_x": 0.7,
        "move_y": 0.4,
        "should_click": true,
        "target_type": "enemy"
      }
    }
  ]
}
```

### Weights

| Quality | Weight | Meaning |
|---------|--------|---------|
| `bad` | 2.0 | Definitely wrong, learn strongly |
| `needs_improvement` | 1.5 | Could be better |
| `good` | 0.5 | Already correct, low priority |

---

## Enhanced VLM (`--enhanced-vlm`)

### How It Works

Three levels of analysis at different frequencies:

| Level | Frequency | What It Analyzes |
|-------|-----------|-----------------|
| **Strategic** | Every 5s | Overall situation: farm/flee/explore? |
| **Tactical** | Every 1s | Target selection, combat tactic |
| **Execution** | Every 0.3s | Was that specific action correct? |

### Correction Format

```json
{
  "source": "enhanced_vlm",
  "corrections": [
    {
      "timestamp": 1234567890.123,
      "level": "tactical",
      "state_vector": [0.1, 0.2, ...],
      "bot_action": {"move_x": 0.5, "move_y": 0.5},
      "vlm_correction": {
        "move_x": 0.7,
        "move_y": 0.3,
        "should_attack": true,
        "tactic": "orbiting",
        "optimal_distance": 0.12
      },
      "vlm_full_result": {
        "priority_target": {"type": "enemy", "name": "Sibelon"},
        "recommended_tactic": "kiting",
        "confidence": 0.85
      },
      "context": {
        "mode": "AGGRESSIVE",
        "health": 0.75,
        "num_enemies": 2
      }
    }
  ]
}
```

### Weights by Level

| Level | Weight | Reasoning |
|-------|--------|-----------|
| `execution` | 2.5 | Immediate action correction - most actionable |
| `tactical` | 2.0 | Target/tactic correction |
| `strategic` | 1.0 | General guidance, less specific |

---

## Bad Stop (`F2` key)

### How It Works

The simplest correction system - no VLM required!

1. **Bot does something wrong** (face-tanks, ignores enemy, etc.)
2. **Press F2** immediately
3. **Last ~2 seconds of actions saved** as negative training data
4. **Bot pauses** so you can resume or stop

### When to Use

- Bot face-tanks a dangerous enemy
- Bot runs away from easy targets
- Bot ignores loot boxes
- Bot clicks on nothing
- Any behavior you want to discourage

### Correction Format

```json
{
  "source": "bad_stop",
  "timestamp": 1234567890,
  "reason": "User pressed F2 to indicate bot behavior was wrong",
  "corrections": [
    {
      "timestamp": 1234567890.123,
      "quality": "bad",
      "state_vector": [0.1, 0.2, ...],
      "bot_action": {
        "move_x": 0.51,
        "move_y": 0.49,
        "clicked": true,
        "ctrl_attack": false
      },
      "correct_action": null,
      "source": "bad_stop",
      "mode": "AGGRESSIVE"
    }
  ]
}
```

### Weight

| Source | Weight | Reasoning |
|--------|--------|-----------|
| `bad_stop` | 3.0 | Highest - user explicitly said this was wrong |

The high weight (3.0) means bad stop corrections have more influence than VLM corrections. You know better than the VLM what's wrong!

---

## Using Corrections in Training

### Basic Training with Corrections

```bash
# Train with your recordings + all VLM corrections
python train.py --corrections
```

### Fine-tune with Corrections Only (Fast)

```bash
# Quick update using only corrections (skip full recordings)
python train.py --finetune --corrections-only
```

### Full Training

```bash
# Full training with more epochs
python train.py --corrections --epochs 20
```

---

## File Locations

| What | Location |
|------|----------|
| Self-improve corrections | `data/vlm_corrections/session_*/corrections.json` |
| Enhanced VLM corrections | `data/vlm_corrections/enhanced_*.json` |
| **Bad stop corrections** | `data/vlm_corrections/bad_stop_*.json` |
| Debug screenshots | `data/vlm_corrections/session_*/debug/` |
| System prompt (editable) | `data/vlm_system_prompt.txt` |

---

## Workflow Example

### 1. Record Your Gameplay

```bash
python -m darkorbit_bot.reasoning.filtered_recorder
# Play the game, press K to save good sequences
```

### 2. Train Initial Model

```bash
python train.py --epochs 10
```

### 3. Run Bot with VLM Watching

```bash
# Option A: Basic self-improve
python -m darkorbit_bot.reasoning.bot_controller --policy models/policy.pt --self-improve

# Option B: Enhanced VLM (richer analysis)
python -m darkorbit_bot.reasoning.bot_controller --policy models/policy.pt --enhanced-vlm
```

### 4. Let Bot Play (VLM generates corrections)

- Bot plays using current policy
- VLM watches and critiques mistakes
- Corrections saved automatically

### 5. Retrain with Corrections

```bash
# Quick fine-tune with just corrections
python train.py --finetune --corrections-only

# Or full retrain including recordings
python train.py --corrections --epochs 5
```

### 6. Repeat

Each cycle improves the model by learning from its mistakes.

---

## Customizing VLM Behavior

Edit `data/vlm_system_prompt.txt` to customize:

- Enemy threat levels and distances
- Combat tactics (orbiting, kiting, etc.)
- Good/bad behavior definitions
- Control mappings (Ctrl=attack, Space=rocket)

The VLM uses this knowledge when critiquing the bot.

---

## Troubleshooting

### No Corrections Generated

1. Check LM Studio is running at `http://localhost:1234`
2. Verify a vision model is loaded (e.g., `qwen/qwen3-vl-8b`)
3. Check `data/vlm_corrections/` for files

### Corrections Not Used in Training

1. Run `python train.py --corrections` (not just `python train.py`)
2. Check corrections have `state_vector` (required for training)
3. Verify files exist in `data/vlm_corrections/`

### Model Not Improving

1. Need enough corrections (aim for 50+ per session)
2. Use `--finetune` to keep existing knowledge
3. Check correction quality in debug JSON files

---

## Technical Details

### State Vector

The `state_vector` is a 128 or 134 float array containing:
- Detection features (enemies, boxes)
- Player state (health, position, velocity)
- Mode (PASSIVE/AGGRESSIVE)
- Movement patterns (if 134-dim)

This is what the neural network sees and what it learns to map to actions.

### Action Vector

8-dimensional output:
```
[0] move_x / aim_x      (0-1, screen position)
[1] move_y / aim_y      (0-1, screen position)
[2] should_click/fire   (0 or 1)
[3] is_enemy            (0 or 1)
[4] distance            (0-1, to target)
[5] ctrl_attack         (0 or 1, Ctrl key)
[6] space_rocket        (0 or 1, Space key)
[7] shift_special       (0 or 1, Shift key)
```

### Weight System

Higher weights = stronger learning signal:
- Recording samples: weight 1.0 (baseline)
- Good corrections: weight 0.5 (already correct)
- Needs improvement: weight 1.5
- Bad corrections: weight 2.0-2.5 (learn strongly)

This prioritizes learning from mistakes over reinforcing already-good behavior.

---

## Meta-Learning (`--meta-learn`)

### Self-Improving VLM System

The meta-learner analyzes VLM performance and suggests improvements to the system prompt.

```bash
# Enable during bot session
python -m darkorbit_bot.reasoning.bot_controller --policy models/policy.pt --enhanced-vlm --meta-learn

# Or run standalone after sessions
python -m darkorbit_bot.reasoning.vlm_meta_learner
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     AFTER BOT SESSION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Collect all VLM corrections from session                    │
│           │                                                     │
│           ▼                                                     │
│  2. Analyze patterns:                                           │
│     - Quality distribution (good/bad/needs_improvement)         │
│     - Combat tactics detected                                   │
│     - Bad stops (user disagreements)                            │
│           │                                                     │
│           ▼                                                     │
│  3. Send to "thinking" LLM for meta-analysis:                   │
│     - What is VLM doing well?                                   │
│     - What is VLM missing?                                      │
│     - Why are users pressing F2 (bad stop)?                     │
│           │                                                     │
│           ▼                                                     │
│  4. Generate improvement suggestions:                           │
│     - Specific prompt changes                                   │
│     - New sections to add                                       │
│     - Priority fixes                                            │
│           │                                                     │
│           ▼                                                     │
│  5. Save to: data/vlm_prompt_suggestions.md                     │
│     Optionally auto-apply with: --apply flag                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Analyzed

| Data Source | What It Tells Us |
|-------------|------------------|
| VLM corrections | Is VLM correctly rating bot actions? |
| Bad stops (F2) | Where does USER disagree with bot/VLM? |
| Combat tactics | Is VLM recommending good tactics? |
| Quality distribution | Is VLM too harsh/lenient? |

### Output Files

| File | Purpose |
|------|---------|
| `data/meta_analysis/meta_analysis_*.json` | Raw analysis data |
| `data/vlm_prompt_suggestions.md` | Human-readable suggestions |
| `data/vlm_system_prompt.txt.backup` | Backup before changes |

### Applying Suggestions

```bash
# Review suggestions first
cat data/vlm_prompt_suggestions.md

# Apply with confirmation prompt
python -m darkorbit_bot.reasoning.vlm_meta_learner --apply

# The system will:
# 1. Show proposed changes
# 2. Ask for confirmation
# 3. Backup current prompt
# 4. Apply changes
```

### The Self-Improvement Loop

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│    ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│    │  Bot     │───▶│   VLM    │───▶│   Training       │     │
│    │  Plays   │    │  Watches │    │   Improves Bot   │     │
│    └──────────┘    └──────────┘    └──────────────────┘     │
│         ▲                │                  │               │
│         │                ▼                  │               │
│         │         ┌──────────────┐          │               │
│         │         │ Meta-Learner │          │               │
│         │         │ Improves VLM │          │               │
│         │         └──────────────┘          │               │
│         │                │                  │               │
│         └────────────────┴──────────────────┘               │
│                                                              │
│              RECURSIVE SELF-IMPROVEMENT                      │
└──────────────────────────────────────────────────────────────┘
```

The bot improves from VLM corrections, and the VLM improves from meta-analysis.
Over time, both get better!
