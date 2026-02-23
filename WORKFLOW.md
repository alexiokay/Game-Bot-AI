# DarkOrbit Bot - Complete Workflow

## Overview

This bot learns to play DarkOrbit by watching YOU play. It has two main phases:

1. **Recording Phase** - Captures your gameplay (mouse, clicks, game state)
2. **Training Phase** - Learns from your recordings to mimic your playstyle

The VLM (Vision Language Model) is an **optional enhancement** that adds quality assessment to training data.

---

## What Each Component Does

### 1. YOLO Detector (`best.pt`)
**Purpose:** Sees the game in real-time

**What it detects:**
- Enemies: Devo, Lordakia, Mordon, Saimon, Sibelon, Struener
- Collectibles: BonusBox
- Player ship, health bar, minimap elements

**Output:** Bounding boxes with class names and positions (x, y, width, height)

---

### 2. Filtered Recorder (`filtered_recorder.py`)
**Purpose:** Records YOUR gameplay with rich context

**What it captures every frame (~30 FPS):**
```
STATE VECTOR (what the bot "sees"):
├── Mouse position (normalized 0-1)
├── Detected objects (up to 10 enemies, 5 boxes)
│   ├── Position (x, y)
│   ├── Size (width, height)
│   └── Distance from mouse
├── Health estimate
├── Mode (PASSIVE/AGGRESSIVE)
└── Click state

ACTION VECTOR (what YOU did):
├── Mouse X, Y (where you moved)
├── Should click (did you click?)
├── Key action (what key pressed)
└── Wait time (pause before action)
```

**Screenshot capture (every 2 seconds):**
- Saves actual game image (640x360 JPEG)
- Saves context JSON alongside:
  ```json
  {
    "mode": "AGGRESSIVE",
    "mouse_pos": [0.52, 0.48],
    "num_enemies": 2,
    "num_boxes": 0,
    "enemy_positions": [[0.3, 0.4], [0.7, 0.6]],
    "recent_clicks": 3,
    "recent_actions": [...]
  }
  ```

**When sequences are saved:**
- On KILL detection (enemy count drops = you killed it)
- Saves last 10 seconds of gameplay leading to the kill

---

### 3. Pattern Analyzer (`analyze_patterns.py`)
**Purpose:** Extracts YOUR personal movement style

**What it analyzes from raw input recordings:**
```
MOVEMENT STYLE:
├── Speed: How fast you move the mouse
├── Curves: Do you move straight or curved?
├── Deceleration: Do you slow down before clicking?

CLICK TIMING:
├── Hold duration: How long you hold mouse button (avg 120ms)
├── Pre-click pause: Do you stop before clicking? (avg 30ms)
├── Post-click pause: Pause after releasing (avg 50ms)
├── Double-click rate: How often you double-click (5%)
```

**Output:** `data/my_movement_profile.json`
- Bot uses this to click and move LIKE YOU
- Makes bot behavior look human, not robotic

---

### 4. VLM Annotator (`vlm_annotator.py`)
**Purpose:** Adds quality assessment to training data

**This is WHERE the "good/bad" comes from!**

**What the VLM receives:**
1. Actual screenshot image (base64 encoded JPEG)
2. Context data:
   ```
   - Mode: AGGRESSIVE
   - Mouse position: (0.52, 0.48)
   - Enemies visible: 2
   - Boxes visible: 0
   - Recent clicks: 3
   - Player was: actively clicking targets
   ```

**What the VLM analyzes (by looking at the image + context):**
```
The VLM SEES the screenshot and UNDERSTANDS:
├── "There are 2 enemy ships on screen"
├── "Player's mouse is near the left enemy"
├── "Player clicked 3 times recently"
├── "But player is clicking the WRONG enemy (farther one)"

VLM OUTPUT:
{
  "situation": "combat",
  "threat": "medium",
  "quality": "needs_improvement",  ← THIS IS THE KEY OUTPUT
  "reasoning": "Player is attacking but targeting the farther enemy
               instead of the closer threat",
  "suggestion": "Click on the closer enemy at position (0.3, 0.4)"
}
```

**Why this matters for training:**
- `quality: "good"` → weight = 1.0 (learn fully from this)
- `quality: "needs_improvement"` → weight = 0.3 (learn less from this)
- Bad examples don't get thrown away, just weighted down

**VLM is NOT:**
- Running in real-time during bot operation (too slow)
- Just returning random good/bad (it analyzes the actual screenshot)
- Required (training works without it, just less refined)

---

### 5. Training (`train.py`)
**Purpose:** Teach the neural network to mimic your gameplay

**Input:** All recorded sequences with states + actions

**How it learns:**
```
For each frame in recording:
    STATE (what was visible) → Neural Network → PREDICTED ACTION
                                                      ↓
                                              Compare with YOUR ACTION
                                                      ↓
                                              Calculate loss, update weights

Loss function:
├── Position loss (MSE): How close is predicted mouse to your mouse?
├── Click loss (BCE): Did it predict click correctly?
└── Weighted by VLM quality (if available)
```

**Output:** `data/checkpoints/policy_latest.pt`
- Bi-LSTM neural network weights
- Separate heads for PASSIVE (collecting) and AGGRESSIVE (combat)

---

### 6. Bot Controller (`bot_controller.py`)
**Purpose:** Play the game using learned behavior

**Real-time loop (30 FPS):**
```
1. SEE
   ├── Capture screen
   ├── Run YOLO detection
   └── Build state vector

2. THINK
   ├── Feed state sequence to trained network
   ├── Get predicted action (mouse position, should click)
   ├── Smart targeting override (snap to detected targets)
   └── Exploration if nothing visible

3. ACT
   ├── Move mouse (using YOUR movement profile for human-like curves)
   ├── Click (using YOUR click timing)
   └── Combat keys (Ctrl for attack, Space for rockets)
```

**Reasoning log shows:**
```
💭 SEE: 2 enemies (Devo, Lordakia) | MODE: AGGRESSIVE | DECIDE: Attack Devo!
💭 SEE: 1 boxes | MODE: PASSIVE (collecting) | DECIDE: Click to collect box
💭 SEE: Nothing | EXPLORE: Idle for 2.3s, searching map...
```

---

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     RECORDING PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   YOU PLAY THE GAME                                              │
│         ↓                                                        │
│   filtered_recorder.py captures:                                 │
│   ├── State vectors (30 FPS)                                     │
│   ├── Your actions (mouse, clicks)                               │
│   └── Screenshots (every 2 sec)                                  │
│         ↓                                                        │
│   Saves to: data/recordings/session_YYYYMMDD_HHMMSS/             │
│   ├── sequence_0001_KILL_AGGRESSIVE.json                         │
│   ├── sequence_0002_KILL_PASSIVE.json                            │
│   └── screenshots/                                               │
│       ├── seq0001_frame00.jpg                                    │
│       └── seq0001_frame00_context.json                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ANALYSIS PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   analyze_patterns.py                                            │
│   ├── Reads raw input recordings                                 │
│   ├── Extracts YOUR movement style                               │
│   └── Saves: data/my_movement_profile.json                       │
│                                                                  │
│   vlm_annotator.py (OPTIONAL, requires LM Studio)                │
│   ├── Loads each sequence                                        │
│   ├── Sends screenshot + context to VLM                          │
│   ├── VLM analyzes: "Was this a good action?"                    │
│   └── Adds quality assessment to sequence JSON                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   train.py                                                       │
│   ├── Loads all SUCCESS sequences                                │
│   ├── Creates sliding window samples (every frame)               │
│   ├── Applies VLM quality weights (if available)                 │
│   ├── Trains Bi-LSTM network                                     │
│   │   ├── PASSIVE head (collecting boxes)                        │
│   │   └── AGGRESSIVE head (combat)                               │
│   └── Saves: data/checkpoints/policy_latest.pt                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PLAYING PHASE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   bot_controller.py                                              │
│   ├── Loads trained policy network                               │
│   ├── Loads YOUR movement profile                                │
│   ├── Real-time loop:                                            │
│   │   ├── YOLO detects game objects                              │
│   │   ├── Network predicts action                                │
│   │   ├── Smart targeting overrides                              │
│   │   └── Executes with human-like timing                        │
│   └── Hotkeys: F1=Pause, F3=Stop, F5=Reasoning                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Self-Improvement System (NEW)

The bot can now learn from its own mistakes using VLM critique.

### How It Works

```
BOT PLAYS THE GAME
       ↓
VLM watches (every 3 seconds)
       ↓
VLM critiques: "Bot is idle, should attack enemy at (0.3, 0.5)"
       ↓
Saves CORRECTION: {state, bot_action, CORRECT_action, quality}
       ↓
Next training includes corrections with high weight
       ↓
Bot improves!
```

### The Key Insight

The VLM doesn't just say "bad" - it tells us **what the correct action should be**.
This gives us new training data with the correct target, not the bot's mistake.

### Correction Weighting

- `bad` actions → weight = 2.0 (definitely wrong, must learn)
- `needs_improvement` → weight = 1.5 (could be better)
- `good` → weight = 0.5 (already correct, less emphasis)

### Integration with Bot

```python
# In bot_controller.py
from reasoning.self_improver import SelfImprover

improver = SelfImprover()
improver.start_session()
improver.start_watching()

# In control loop:
improver.submit_frame(frame, {
    'bot_action': action,
    'detections': detections,
    'mode': mode,
    'idle_time': self.idle_time,
    'state_vector': state_sequence[-1].tolist()
})

# On stop:
improver.stop_watching()
```

### Training with Corrections

```bash
# Normal training
python train.py --epochs 20

# Training WITH VLM corrections from self-improvement
python train.py --epochs 20 --corrections
```

---

## Commands Reference

### Step 1: Record Gameplay
```bash
.\.venv\Scripts\python.exe darkorbit_bot/reasoning/filtered_recorder.py --monitor 1
```
- Play normally, kill enemies, collect boxes
- Press `Q` to stop
- Creates: `data/recordings/session_YYYYMMDD_HHMMSS/`

### Step 2: Analyze Your Movement Style
```bash
.\.venv\Scripts\python.exe darkorbit_bot/analysis/analyze_patterns.py
```
- No arguments needed
- Creates: `data/my_movement_profile.json`

### Step 3: VLM Annotation (Optional)
```bash
# Requires LM Studio running with a vision model
.\.venv\Scripts\python.exe darkorbit_bot/reasoning/vlm_annotator.py --session latest
```
- Sends real screenshots to VLM for quality analysis
- Updates sequence JSON files with `vlm_context`

### Step 4: Train the Network
```bash
.\.venv\Scripts\python.exe darkorbit_bot/train.py --epochs 20
```
- More epochs = better learning (but diminishing returns)
- Creates: `data/checkpoints/policy_latest.pt`

### Step 5: Run the Bot
```bash
.\.venv\Scripts\python.exe darkorbit_bot/reasoning/bot_controller.py --policy data/checkpoints/policy_latest.pt --monitor 1
```
- F1 = Pause/Resume
- F3 = Emergency Stop
- F5 = Toggle reasoning log

---

## What Makes the VLM Analysis Useful

Without VLM:
- All recorded kills are treated equally
- Bot learns from both good and bad plays

With VLM:
- VLM sees the actual screenshot
- Understands context (enemies, positioning, what you clicked)
- Evaluates: "Was clicking that target optimal?"
- Bad plays get lower weight in training

**Example VLM reasoning:**
```
Screenshot shows: 2 enemies, player clicking the far one
VLM says: "needs_improvement - should click closer enemy first"

Screenshot shows: 1 enemy, player clicking directly on it
VLM says: "good - efficient target acquisition"

Screenshot shows: boxes nearby, player collecting them
VLM says: "good - appropriate passive behavior"
```

The VLM doesn't just return random good/bad - it actually analyzes the visual scene and player actions to make that determination.
