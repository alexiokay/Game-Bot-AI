# DarkOrbit Bot - AI Architecture Documentation

## Table of Contents
1. [Current Architecture](#current-architecture)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Analysis](#component-analysis)
4. [Limitations](#limitations)
5. [Modern AI Approaches](#modern-ai-approaches)
6. [Recommended Upgrades](#recommended-upgrades)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Current Architecture

### Overview

The bot uses a **Bi-directional LSTM (Long Short-Term Memory) Policy Network** with dual behavioral heads. This is a sequence-to-action model that processes a window of recent game states and outputs control actions.

### Key Specifications

| Component | Specification |
|-----------|--------------|
| **Model Type** | Bi-LSTM with Attention |
| **Input** | 50 frames × 128 features = 6,400 values |
| **Hidden Size** | 256 units per direction (512 total) |
| **Layers** | 2 LSTM layers |
| **Output Heads** | 2 (Passive + Aggressive) |
| **Output Size** | 5 actions per head |
| **Inference Time** | ~5ms on GPU |
| **Parameters** | ~2.5M |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT ARCHITECTURE: Bi-LSTM Policy                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT: State Sequence (50 frames × 128 features)                          │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │  Frame 1  │  Frame 2  │  ...  │  Frame 49  │  Frame 50   │             │
│   │  [128]    │  [128]    │  ...  │  [128]     │  [128]      │             │
│   └───────────────────────────────────────────────────────────┘             │
│                              │                                               │
│                              ▼                                               │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │              Bi-LSTM (2 layers, 256 hidden)               │             │
│   │                                                           │             │
│   │    ←←← Forward LSTM ←←←    →→→ Backward LSTM →→→         │             │
│   │           256                      256                    │             │
│   └───────────────────────────────────────────────────────────┘             │
│                              │                                               │
│                         [512 features]                                       │
│                              ▼                                               │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │                  Attention Layer                          │             │
│   │     "Which frames are most important?"                    │             │
│   │     512 → 256 → 1 → softmax weights                       │             │
│   └───────────────────────────────────────────────────────────┘             │
│                              │                                               │
│                    Weighted Context [512]                                    │
│                              │                                               │
│              ┌───────────────┴───────────────┐                              │
│              ▼                               ▼                              │
│   ┌─────────────────────┐       ┌─────────────────────┐                    │
│   │    PASSIVE HEAD     │       │   AGGRESSIVE HEAD   │                    │
│   │  512→128→64→5      │       │    512→128→64→5     │                    │
│   │                     │       │                     │                    │
│   │  • move_x (0-1)    │       │  • aim_x (0-1)      │                    │
│   │  • move_y (0-1)    │       │  • aim_y (0-1)      │                    │
│   │  • should_click    │       │  • should_fire      │                    │
│   │  • key_action      │       │  • ability_key      │                    │
│   │  • wait_time       │       │  • dodge_direction  │                    │
│   └─────────────────────┘       └─────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE DATA PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  SCREEN CAPTURE              YOLO DETECTION              STATE BUILDER        │
│  ┌──────────┐               ┌──────────────┐            ┌──────────────┐     │
│  │ Monitor  │ ────────────▶ │  best.pt     │ ─────────▶ │ 128-dim      │     │
│  │ 1920×1080│    frame      │  YOLOv8      │  objects   │ state vector │     │
│  └──────────┘               └──────────────┘            └──────────────┘     │
│                                    │                           │              │
│                                    ▼                           │              │
│                             Detected Objects:                  │              │
│                             • Enemies (Devo, Lordakia, etc.)   │              │
│                             • BonusBox (loot)                  │              │
│                             • Player ship                      │              │
│                             • Health bar                       │              │
│                                                                │              │
│                                                                ▼              │
│  STATE VECTOR COMPONENTS (128 features):                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ [0-9]   Player: x, y, health, shield, speed, ...                       │  │
│  │ [10-49] Enemies: up to 8 enemies × 5 features (x,y,dist,class,threat)  │  │
│  │ [50-69] Boxes: up to 4 boxes × 5 features (x,y,dist,type,priority)     │  │
│  │ [70-89] Combat: is_attacking, time_since_kill, damage_taken, ...       │  │
│  │ [90-127] Context: mode, idle_time, exploration_target, ...             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                                    ▼                                          │
│  SEQUENCE BUFFER (Last 50 frames @ 30 FPS = ~1.7 seconds of history)         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ [Frame t-49] [Frame t-48] ... [Frame t-1] [Frame t (current)]          │  │
│  │    [128]        [128]    ...    [128]        [128]                     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                                    ▼                                          │
│                          BI-LSTM POLICY NETWORK                               │
│                                    │                                          │
│                                    ▼                                          │
│                             ACTION OUTPUT                                     │
│                        (x, y, click, key, wait)                               │
│                                    │                                          │
│                                    ▼                                          │
│                          HUMANIZED EXECUTION                                  │
│                    ┌──────────────────────────────┐                          │
│                    │ • Bezier curve mouse paths   │                          │
│                    │ • Variable click timing      │                          │
│                    │ • Natural pauses             │                          │
│                    │ • Occasional overshoot       │                          │
│                    └──────────────────────────────┘                          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Analysis

### 1. State Builder (`state_builder.py`)

Converts raw YOLO detections into a fixed-size state vector.

```python
State Vector [128 features]:
├── Player State [10]
│   ├── position (x, y)           # Normalized 0-1
│   ├── health                    # 0-1
│   ├── shield                    # 0-1
│   └── velocity, orientation...
├── Enemy Features [40]
│   └── Up to 8 enemies × 5 features each
│       ├── position (x, y)
│       ├── distance to player
│       ├── class encoding
│       └── threat level
├── Loot Features [20]
│   └── Up to 4 boxes × 5 features each
├── Combat State [20]
│   ├── is_attacking
│   ├── time_since_last_kill
│   ├── damage_taken_rate
│   └── ...
└── Context [38]
    ├── mode (PASSIVE/AGGRESSIVE)
    ├── idle_time
    └── exploration state
```

### 2. Policy Network (`policy_network.py`)

```python
class DualHeadPolicy(nn.Module):
    """
    Bi-LSTM with attention and dual output heads.

    Architecture:
    - Input: (batch, seq_len=50, features=128)
    - LSTM: 2 layers, 256 hidden, bidirectional
    - Attention: Learned importance weights per frame
    - Heads: Separate MLPs for passive/aggressive
    """

    def __init__(self):
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        self.passive_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        self.aggressive_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
```

### 3. Action Outputs

**Passive Head** (for looting, exploring):
| Output | Range | Description |
|--------|-------|-------------|
| move_x | 0-1 | Target X position (normalized) |
| move_y | 0-1 | Target Y position (normalized) |
| should_click | binary | Whether to click |
| key_action | 0-17 | Which key to press (if any) |
| wait_time | 0-1 | How long to wait before acting |

**Aggressive Head** (for combat):
| Output | Range | Description |
|--------|-------|-------------|
| aim_x | 0-1 | Target X to aim at |
| aim_y | 0-1 | Target Y to aim at |
| should_fire | binary | Whether to attack |
| ability_key | 0-17 | Which ability to use |
| dodge_direction | -1 to 1 | Evasion direction |

---

## Limitations

### Current Problems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT LIMITATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. NO EXPLICIT GOALS                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Current: See enemy → Output action → Forget                      │     │
│     │ Problem: Can't track "I'm hunting enemy #1"                      │     │
│     │          Each frame is independent decision                      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  2. NO OBJECT PERMANENCE                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Current: Enemy at (0.3, 0.5) this frame                         │     │
│     │ Next:    Enemy at (0.31, 0.52) - Is it the SAME enemy?          │     │
│     │ Problem: No tracking across frames, treats as new object        │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  3. PURELY REACTIVE                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Current: state → action (immediate reaction)                     │     │
│     │ Problem: Can't plan "first kill A, then collect box B"          │     │
│     │          No lookahead or consequence prediction                  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  4. CAN'T EXPLAIN DECISIONS                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Current: Black box neural network                                │     │
│     │ Problem: Why did it click there? Unknown.                        │     │
│     │          Hard to debug or improve                                │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  5. LIMITED TEMPORAL REASONING                                               │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Current: 50 frames (~1.7s) context window                        │     │
│     │ Problem: Can't remember "5 seconds ago I saw a rare box"        │     │
│     │          No long-term memory of the game session                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Behavioral Symptoms

| Symptom | Root Cause |
|---------|------------|
| Bot switches targets randomly | No goal persistence |
| Bot "forgets" enemies that move | No object tracking |
| Bot doesn't chase fleeing enemies | No planning |
| Bot makes suboptimal decisions | Purely reactive |
| Hard to understand why bot did X | Black box model |

---

## Modern AI Approaches

### Option 1: Transformer + Goal Memory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTION 1: TRANSFORMER + GOAL MEMORY                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GOAL MANAGER                    STATE ENCODER                               │
│  ┌──────────────┐               ┌──────────────────────────────┐            │
│  │ Active Goals │               │     Transformer Encoder      │            │
│  │ ────────────│               │                              │            │
│  │ • Kill enemy │               │  Self-Attention across       │            │
│  │   at (0.3,0.5)│              │  ALL detected objects        │            │
│  │ • Collect box │              │                              │            │
│  │   at (0.7,0.2)│              │  Object 1 ←→ Object 2 ←→ ... │            │
│  │ • Explore NE  │              │       ↕          ↕           │            │
│  │              │               │  Object 3 ←→ Object 4 ←→ ... │            │
│  └──────────────┘               └──────────────────────────────┘            │
│        │                                    │                               │
│        └──────────────┬─────────────────────┘                               │
│                       ▼                                                      │
│        ┌──────────────────────────────┐                                     │
│        │      Cross-Attention         │                                     │
│        │  "Which goal applies to      │                                     │
│        │   which detected object?"    │                                     │
│        └──────────────────────────────┘                                     │
│                       │                                                      │
│                       ▼                                                      │
│        ┌──────────────────────────────┐                                     │
│        │      Action Decoder          │                                     │
│        │  Output: (x, y, click, goal) │                                     │
│        └──────────────────────────────┘                                     │
│                                                                              │
│  BENEFITS:                                                                   │
│  ✓ Explicit goal tracking (remembers "I'm hunting enemy #1")                │
│  ✓ Object-centric attention (focus on specific targets)                     │
│  ✓ Can switch goals dynamically                                             │
│  ✓ Better generalization                                                    │
│                                                                              │
│  COST: More complex, ~10-20ms inference                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Option 2: World Model + Planning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTION 2: WORLD MODEL + PLANNING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                        WORLD MODEL                               │        │
│  │  "If I do X, what happens next?"                                │        │
│  │                                                                  │        │
│  │  Current State → Predicted Future States (3-5 steps)            │        │
│  │       ↓              ↓            ↓          ↓                  │        │
│  │    [now]  →  [+0.5s]  →  [+1s]  →  [+1.5s]  →  [+2s]           │        │
│  │                                                                  │        │
│  │  Examples:                                                       │        │
│  │  • "If I click enemy, it will die in 2s"                        │        │
│  │  • "If I move left, box will be reachable"                      │        │
│  │  • "If I stay, enemy will reach me"                             │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                     PLANNING MODULE                              │        │
│  │                                                                  │        │
│  │  Evaluate future outcomes → Pick best action sequence            │        │
│  │                                                                  │        │
│  │  Action A → [future] → Score: 0.8 (kills enemy)                 │        │
│  │  Action B → [future] → Score: 0.3 (enemy escapes)               │        │
│  │  Action C → [future] → Score: 0.5 (collect box instead)         │        │
│  │                                                                  │        │
│  │  → Execute Action A                                              │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  BENEFITS:                                                                   │
│  ✓ Can PLAN ahead (not just react)                                          │
│  ✓ Understands consequences                                                  │
│  ✓ Better decision making                                                    │
│  ✓ State-of-the-art in game AI (DreamerV3, IRIS)                            │
│                                                                              │
│  COST: Complex to train, ~50-100ms planning per decision                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Option 3: Hybrid - Fast Reflex + Slow Reasoning (RECOMMENDED)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               OPTION 3: HYBRID - FAST REFLEX + SLOW REASONING               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  This is the RECOMMENDED approach for this use case.                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │           FAST PATH (current Bi-LSTM, ~5ms)                     │        │
│  │                                                                  │        │
│  │  State → Bi-LSTM → Action (reflexive, low-level)               │        │
│  │                                                                  │        │
│  │  Handles: Aiming, clicking, immediate reactions                 │        │
│  └────────────────────────────┬────────────────────────────────────┘        │
│                               │                                              │
│                               │  Modulated by ↓                              │
│                               │                                              │
│  ┌────────────────────────────▼────────────────────────────────────┐        │
│  │           SLOW PATH (Goal Tracker, ~50ms, runs async)           │        │
│  │                                                                  │        │
│  │  ┌────────────────┐     ┌────────────────┐                      │        │
│  │  │ Object Tracker │ →   │  Goal Manager  │                      │        │
│  │  │                │     │                │                      │        │
│  │  │ Enemy #1: alive│     │ PRIMARY GOAL:  │                      │        │
│  │  │ Enemy #2: dead │     │ Kill Enemy #1  │                      │        │
│  │  │ Box #1: nearby │     │                │                      │        │
│  │  │ Box #2: far    │     │ FALLBACK:      │                      │        │
│  │  └────────────────┘     │ Collect Box #1 │                      │        │
│  │                         └────────────────┘                      │        │
│  │                                                                  │        │
│  │  Outputs: GOAL_TARGET, GOAL_TYPE, PRIORITY                      │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    INTEGRATION LAYER                             │        │
│  │                                                                  │        │
│  │  Fast Action + Goal Context → Final Action                      │        │
│  │                                                                  │        │
│  │  • If goal_target visible: bias attention toward it             │        │
│  │  • If goal completed: switch to next goal                       │        │
│  │  • If goal impossible: cancel and re-evaluate                   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  BENEFITS:                                                                   │
│  ✓ Keeps FAST reactions (essential for combat)                              │
│  ✓ Adds GOAL PERSISTENCE (tracks specific targets)                          │
│  ✓ DYNAMIC SWITCHING (can change mind mid-action)                           │
│  ✓ Easier to implement incrementally                                        │
│  ✓ VLM can influence goal selection (self-improvement!)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recommended Upgrades

### Phase 1: Goal-Conditioned Policy (Minimal Change)

Keep the existing Bi-LSTM but add goal information to the input.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED ARCHITECTURE                             │
│                     "Goal-Conditioned Bi-LSTM Policy"                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DETECTIONS                    OBJECT TRACKER                                │
│  ┌───────────┐                ┌─────────────────────┐                       │
│  │ Enemy #1  │ ──────────────▶│  Track objects      │                       │
│  │ Enemy #2  │                │  across frames      │                       │
│  │ Box #1    │                │                     │                       │
│  └───────────┘                │  Assigns IDs:       │                       │
│                               │  • obj_001 (enemy)  │                       │
│                               │  • obj_002 (enemy)  │                       │
│                               │  • obj_003 (box)    │                       │
│                               └──────────┬──────────┘                       │
│                                          │                                   │
│                                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                       GOAL SELECTOR                              │        │
│  │                                                                  │        │
│  │  Rules + Learning:                                              │        │
│  │  1. If enemies visible → goal = ATTACK closest enemy            │        │
│  │  2. If no enemies, boxes visible → goal = COLLECT closest box   │        │
│  │  3. If nothing visible → goal = EXPLORE random direction        │        │
│  │  4. If under attack → goal = FLEE or COUNTERATTACK              │        │
│  │                                                                  │        │
│  │  Current Goal: ATTACK obj_001 at (0.3, 0.5)                     │        │
│  │  Goal Embedding: [32-dim learned vector]                        │        │
│  └────────────────────────────┬────────────────────────────────────┘        │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                  ENHANCED STATE VECTOR                           │        │
│  │                                                                  │        │
│  │  Original [128] + Goal Context [32] + Target Info [16] = [176]  │        │
│  │                                                                  │        │
│  │  Goal Context [32]:                                             │        │
│  │  • goal_type: [1,0,0,0] = ATTACK (one-hot, 4 dims)             │        │
│  │  • target_x: 0.3                                                │        │
│  │  • target_y: 0.5                                                │        │
│  │  • target_distance: 0.25                                        │        │
│  │  • target_class: enemy type encoding                            │        │
│  │  • time_on_goal: 2.5s (how long pursuing this goal)            │        │
│  │  • goal_confidence: 0.9                                         │        │
│  │  • target_velocity: (vx, vy)                                    │        │
│  │  • ... [32 total features]                                      │        │
│  └────────────────────────────┬────────────────────────────────────┘        │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Bi-LSTM Policy (same as before)               │        │
│  │                    But now GOAL-CONDITIONED!                     │        │
│  │                                                                  │        │
│  │  Input: [176] per frame (includes goal info)                    │        │
│  │  The network learns: "given THIS goal, do THIS action"          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  KEY INSIGHT: Same architecture, just RICHER INPUT                          │
│  The goal info tells the network WHAT you want to achieve,                  │
│  the network learns HOW to achieve it.                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Enhanced State Vector Design

```python
# Current state vector: 128 features
current_state = [
    # ... existing 128 features ...
]

# New goal context: 32 features
goal_context = [
    # Goal type (one-hot, 4 dims)
    goal_is_attack,      # 1 if attacking
    goal_is_collect,     # 1 if collecting loot
    goal_is_explore,     # 1 if exploring
    goal_is_flee,        # 1 if fleeing

    # Target information (12 dims)
    target_x,            # 0-1, target position
    target_y,            # 0-1
    target_distance,     # 0-1, normalized distance
    target_vx,           # velocity x
    target_vy,           # velocity y
    target_health,       # if enemy, their health (0-1)
    target_threat,       # threat level (0-1)
    target_value,        # value/priority (0-1)
    target_class_enc,    # 4-dim class embedding

    # Goal state (8 dims)
    time_on_goal,        # seconds pursuing this goal
    goal_progress,       # 0-1, how close to completion
    goal_confidence,     # 0-1, certainty this is right goal
    goal_interrupt_score,# 0-1, should we switch goals?
    prev_goal_type,      # what was previous goal (4 dims)

    # Tactical context (8 dims)
    enemies_in_range,    # count
    boxes_in_range,      # count
    nearest_threat_dist, # distance to closest danger
    escape_route_clear,  # 0-1, can we flee if needed?
    ammo_status,         # 0-1
    cooldown_status,     # 0-1
    ...
]

# Combined: 128 + 32 = 160 features per frame
enhanced_state = current_state + goal_context
```

---

## Implementation Roadmap

### Priority Matrix

| Feature | Complexity | Impact | Priority |
|---------|------------|--------|----------|
| **Object Tracking** | Medium | HIGH | 1st |
| **Goal State** | Low | HIGH | 2nd |
| **Goal Persistence** | Low | MEDIUM | 3rd |
| **Transformer Encoder** | High | MEDIUM | 4th |
| **World Model** | Very High | HIGH | Future |

### Phase 1: Object Tracking (Priority 1)

**Goal**: Track objects across frames so we know "enemy #1" is the same enemy.

```python
# New file: darkorbit_bot/tracking/object_tracker.py

class ObjectTracker:
    """
    Tracks detected objects across frames using IoU matching.
    Assigns persistent IDs to objects.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.tracks = {}  # id -> TrackState
        self.next_id = 1
        self.max_age = max_age  # frames before track is lost
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Match new detections to existing tracks.
        Create new tracks for unmatched detections.
        """
        # 1. Predict where existing tracks should be
        # 2. Match detections to tracks using IoU
        # 3. Update matched tracks
        # 4. Create new tracks for unmatched detections
        # 5. Remove stale tracks

        return tracked_objects

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)
```

**Implementation Steps**:
1. Create `ObjectTracker` class with IoU-based matching
2. Integrate into `bot_controller.py` main loop
3. Store track IDs in state vector
4. Test tracking accuracy

### Phase 2: Goal Manager (Priority 2)

**Goal**: Decide what the bot should be trying to do.

```python
# New file: darkorbit_bot/reasoning/goal_manager.py

@dataclass
class Goal:
    goal_type: str  # "ATTACK", "COLLECT", "EXPLORE", "FLEE"
    target_id: Optional[int]  # Track ID of target
    target_position: Tuple[float, float]
    priority: float
    created_at: float

class GoalManager:
    """
    Manages bot goals and decides priorities.
    """

    def __init__(self):
        self.current_goal: Optional[Goal] = None
        self.goal_history: List[Goal] = []

    def update(self, tracked_objects: List[TrackedObject],
               player_state: Dict) -> Goal:
        """
        Evaluate situation and decide current goal.
        """
        # Priority rules:
        # 1. FLEE if health critical and enemies present
        # 2. ATTACK if enemies visible and health OK
        # 3. COLLECT if boxes visible and no enemies
        # 4. EXPLORE if nothing visible

        new_goal = self._select_best_goal(tracked_objects, player_state)

        # Goal persistence: don't switch too often
        if self._should_keep_current_goal(new_goal):
            return self.current_goal

        self.current_goal = new_goal
        return new_goal

    def _should_keep_current_goal(self, new_goal: Goal) -> bool:
        """
        Prevent erratic goal switching.
        Keep current goal unless:
        - Target died/disappeared
        - Much better opportunity appeared
        - Goal completed
        """
        if self.current_goal is None:
            return False

        # Keep attacking same enemy unless dead
        if (self.current_goal.goal_type == "ATTACK" and
            new_goal.goal_type == "ATTACK" and
            self.current_goal.target_id is not None):
            # Check if current target still exists
            # ...
            return True

        return False
```

### Phase 3: Enhanced State Vector (Priority 3)

**Goal**: Include goal information in the state so the network learns goal-conditioned behavior.

```python
# Modify: darkorbit_bot/reasoning/state_builder.py

class GoalConditionedStateBuilder(StateBuilder):
    """
    Builds state vectors that include goal context.
    """

    def __init__(self):
        super().__init__()
        self.goal_encoder = GoalEncoder()

    def build_state(self, detections, tracked_objects, current_goal) -> np.ndarray:
        # Original state
        base_state = super().build_state(detections)  # [128]

        # Goal context
        goal_features = self.goal_encoder.encode(current_goal, tracked_objects)  # [32]

        # Combined
        return np.concatenate([base_state, goal_features])  # [160]

class GoalEncoder:
    """Encodes goal into fixed-size vector."""

    def encode(self, goal: Goal, tracked_objects: List) -> np.ndarray:
        features = np.zeros(32, dtype=np.float32)

        # Goal type one-hot [0:4]
        goal_types = ["ATTACK", "COLLECT", "EXPLORE", "FLEE"]
        if goal and goal.goal_type in goal_types:
            features[goal_types.index(goal.goal_type)] = 1.0

        # Target info [4:16]
        if goal and goal.target_position:
            features[4] = goal.target_position[0]  # x
            features[5] = goal.target_position[1]  # y
            # ... more target features

        # Goal state [16:24]
        if goal:
            features[16] = min(goal.time_alive() / 10.0, 1.0)  # time on goal
            features[17] = goal.priority
            # ...

        return features
```

### Phase 4: Retrain with Goals (Priority 4)

**Goal**: Train the model to understand goal-conditioned behavior.

```python
# Modify: darkorbit_bot/train.py

def train_goal_conditioned(epochs: int = 20):
    """
    Train with goal context.

    Key change: During recording, we also record what goal
    the human player was pursuing (inferred from actions).
    """

    # Load sequences WITH goal annotations
    samples = load_sequences_with_goals()

    # Create goal-conditioned model
    policy = create_policy(state_size=160)  # 128 + 32 goal features

    # Training loop (same as before, just bigger input)
    for epoch in range(epochs):
        for batch in samples:
            states = batch['states']  # Now [batch, seq, 160]
            actions = batch['actions']

            pred = policy(states, mode=batch['mode'])
            loss = compute_loss(pred, actions)
            # ...
```

### Recording with Goals

To train a goal-conditioned model, we need to annotate recordings with goals:

```python
# Modify: filtered_recorder.py

class GoalInferrer:
    """
    Infers what goal the human player is pursuing from their actions.
    """

    def infer_goal(self, recent_frames: List[BufferFrame],
                   detections: List) -> Goal:
        # Analyze recent mouse movements and clicks
        # If clicking on enemies → ATTACK goal
        # If clicking on boxes → COLLECT goal
        # If moving to empty space → EXPLORE goal
        # If moving away from enemies → FLEE goal

        click_targets = self._analyze_click_targets(recent_frames, detections)

        if click_targets.get('enemy'):
            return Goal(
                goal_type="ATTACK",
                target_id=click_targets['enemy'].track_id,
                target_position=(click_targets['enemy'].x, click_targets['enemy'].y),
                priority=1.0
            )
        # ... etc
```

---

## Future: Advanced Architectures

### Transformer-Based Policy

For even better performance, replace LSTM with Transformer:

```python
class TransformerPolicy(nn.Module):
    """
    Transformer-based policy network.

    Advantages over LSTM:
    - Better long-range dependencies
    - Parallel processing (faster training)
    - Object-level attention
    """

    def __init__(self, state_size=160, num_heads=8, num_layers=4):
        super().__init__()

        self.state_encoder = nn.Linear(state_size, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.passive_head = nn.Linear(256, 5)
        self.aggressive_head = nn.Linear(256, 5)

    def forward(self, state_sequence, mode="PASSIVE"):
        # Encode states
        x = self.state_encoder(state_sequence)

        # Self-attention across time
        x = self.transformer(x)

        # Use last state for action
        context = x[:, -1, :]

        if mode == "AGGRESSIVE":
            return self.aggressive_head(context)
        return self.passive_head(context)
```

### World Model (Future)

For true planning capability:

```python
class WorldModel(nn.Module):
    """
    Predicts future states given current state and action.
    Enables planning by imagining consequences.
    """

    def __init__(self):
        self.state_encoder = nn.Sequential(...)
        self.dynamics = nn.LSTM(...)  # Predicts next state
        self.reward_predictor = nn.Sequential(...)  # Predicts value

    def imagine(self, state, actions, horizon=5):
        """
        Imagine future states for a sequence of actions.
        """
        states = [state]
        rewards = []

        for action in actions:
            next_state = self.dynamics(states[-1], action)
            reward = self.reward_predictor(next_state)
            states.append(next_state)
            rewards.append(reward)

        return states, rewards
```

---

## Summary

### Current State
- **Model**: Bi-LSTM with attention
- **Strengths**: Fast inference, learns from recordings
- **Weaknesses**: No goals, no tracking, purely reactive

### Recommended Upgrade Path

1. **Phase 1**: Add object tracking (assign IDs to detections)
2. **Phase 2**: Add goal manager (decide current objective)
3. **Phase 3**: Enhance state vector (include goal info)
4. **Phase 4**: Retrain with goal-conditioned data

### Expected Improvements

| Metric | Before | After (Est.) |
|--------|--------|--------------|
| Target persistence | Random switching | Tracks until dead |
| Goal completion rate | ~40% | ~70% |
| Decision explainability | None | "Attacking enemy #1" |
| Reaction time | ~50ms | ~50ms (unchanged) |
| Dynamic behavior | Limited | Can switch goals |

---

## References

- **DreamerV3**: World models for game AI (Hafner et al., 2023)
- **IRIS**: Imagination-based RL (Micheli et al., 2023)
- **Attention Is All You Need**: Transformer architecture (Vaswani et al., 2017)
- **SORT/DeepSORT**: Object tracking algorithms
