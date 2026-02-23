# DarkOrbit Bot V2 - Hierarchical Temporal Architecture

## Overview

V2 is a complete redesign based on 2026 state-of-the-art game AI principles. It coexists with V1 - your current Bi-LSTM model remains untouched.

**Key Insight:** Different decisions need different speeds.

| Decision Type | Example | Frequency | Model |
|--------------|---------|-----------|-------|
| Strategy | "Fight or flee?" | 1 Hz | Transformer |
| Tactics | "Which target?" | 10 Hz | Cross-Attention |
| Execution | "Exact mouse pos" | 60 Hz | Mamba (SSM) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DARKORBIT BOT V2 ARCHITECTURE                         │
│                     "Hierarchical Temporal System"                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  LAYER 0: PERCEPTION (every frame, ~12ms total)                        │ │
│  │  ══════════════════════════════════════════                            │ │
│  │                                                                         │ │
│  │  ┌──────────┐      ┌───────────┐      ┌──────────────┐                 │ │
│  │  │   YOLO   │ ───▶ │ ByteTrack │ ───▶ │ State        │                 │ │
│  │  │  (10ms)  │      │   (2ms)   │      │ Encoder      │                 │ │
│  │  └──────────┘      └───────────┘      └──────────────┘                 │ │
│  │                                                                         │ │
│  │  Output: Tracked objects with PERSISTENT IDs                           │ │
│  │  - enemy_001: {x, y, vx, vy, type, hp, age}                           │ │
│  │  - box_003: {x, y, type, value, age}                                  │ │
│  │  - player: {x, y, hp, shield, velocity}                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  LAYER 1: STRATEGIST (every 1-2 seconds, ~20ms)                        │ │
│  │  ══════════════════════════════════════════════                        │ │
│  │                                                                         │ │
│  │  Architecture: Transformer (4 layers, 256 dim, 4 heads)                │ │
│  │                                                                         │ │
│  │  Input: Last 60 seconds of state summaries (downsampled)               │ │
│  │  Output:                                                                │ │
│  │    - Goal embedding (32-64 dim continuous vector)                      │ │
│  │    - Discrete mode: FIGHT | LOOT | FLEE | EXPLORE | CAUTIOUS           │ │
│  │                                                                         │ │
│  │  This replaces the rigid PASSIVE/AGGRESSIVE binary switch.             │ │
│  │  The goal embedding is a RICH representation of intent.                │ │
│  │                                                                         │ │
│  │  Example outputs:                                                       │ │
│  │  - "Aggressively chase that specific low-HP enemy"                     │ │
│  │  - "Defensively collect boxes while watching for threats"              │ │
│  │  - "Flee toward the portal at top-right"                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              │ Goal Embedding (32-64 dim)                    │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  LAYER 2: TACTICIAN (every ~100ms, ~8ms)                               │ │
│  │  ═══════════════════════════════════════════                           │ │
│  │                                                                         │ │
│  │  Architecture: Small Transformer with Cross-Attention (2 layers)       │ │
│  │                                                                         │ │
│  │  Input:                                                                 │ │
│  │    - Current tracked objects (set of N objects)                        │ │
│  │    - Goal embedding from Strategist                                    │ │
│  │    - Last 1-2 seconds of state history                                 │ │
│  │                                                                         │ │
│  │  Cross-Attention Mechanism:                                            │ │
│  │    Query: Goal embedding                                               │ │
│  │    Keys/Values: Object embeddings                                      │ │
│  │    → Learns "given FIGHT goal, which object to target?"               │ │
│  │                                                                         │ │
│  │  Output:                                                                │ │
│  │    - Target selection (attention weights over objects)                 │ │
│  │    - Approach vector: [vx, vy, urgency, aggression]                   │ │
│  │                                                                         │ │
│  │  This is EXPLICIT target selection - you can inspect/override it.      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              │ Target info + Approach vector                 │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  LAYER 3: EXECUTOR (every frame 60fps, ~3ms)                           │ │
│  │  ═══════════════════════════════════════════                           │ │
│  │                                                                         │ │
│  │  Architecture: MAMBA (State Space Model)                               │ │
│  │                                                                         │ │
│  │  Why Mamba instead of LSTM?                                            │ │
│  │    - O(1) inference per step (LSTM is O(n) for attention)             │ │
│  │    - Better long-range dependencies                                    │ │
│  │    - ~3ms inference on RTX 3080                                       │ │
│  │    - 2024+ state-of-the-art for sequence modeling                     │ │
│  │                                                                         │ │
│  │  Input:                                                                 │ │
│  │    - Current state (from perception)                                   │ │
│  │    - Goal embedding (from Strategist)                                  │ │
│  │    - Target info (from Tactician)                                      │ │
│  │    - Hidden state (maintained across frames)                           │ │
│  │                                                                         │ │
│  │  Output:                                                                │ │
│  │    - mouse_x, mouse_y (normalized 0-1)                                │ │
│  │    - should_click (binary)                                            │ │
│  │    - ability_key (discrete)                                           │ │
│  │                                                                         │ │
│  │  The Executor doesn't need to "understand" strategy.                   │ │
│  │  It just executes: "Move toward target X with urgency Y"              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  OUTPUT: Humanized Action Execution                                    │ │
│  │  ═════════════════════════════════════                                 │ │
│  │                                                                         │ │
│  │  Same as V1: Bezier curves, variable timing, natural pauses            │ │
│  │  (Reuse existing humanization code)                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ASYNC: VLM Critique (every 3 seconds, non-blocking)                   │ │
│  │  ═══════════════════════════════════════════════════                   │ │
│  │                                                                         │ │
│  │  Same VLM system but provides RICHER corrections:                      │ │
│  │    - Mode corrections: "should have fled, not fought"                  │ │
│  │    - Target corrections: "should have targeted bomber, not fighter"   │ │
│  │    - Timing corrections: "ability used too late"                       │ │
│  │    - Approach corrections: "should have flanked, not charged"         │ │
│  │                                                                         │ │
│  │  Corrections feed back to train ALL layers:                            │ │
│  │    - Strategist learns better mode selection                           │ │
│  │    - Tactician learns better target prioritization                    │ │
│  │    - Executor learns better motor control                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Latency Budget

| Component | Frequency | Latency | Amortized per Frame |
|-----------|-----------|---------|---------------------|
| YOLO | 60 fps | 10ms | 10ms |
| ByteTrack | 60 fps | 2ms | 2ms |
| Executor (Mamba) | 60 fps | 3ms | 3ms |
| Tactician | 10 fps | 8ms | 0.8ms |
| Strategist | 1 fps | 20ms | 0.3ms |
| **Total** | - | - | **~16ms** |

Well under the 50ms target for responsive gameplay!

---

## Component Specifications

### 1. ByteTrack Object Tracker

**Purpose:** Assign persistent IDs to detected objects across frames.

**Why critical:** Without tracking, the bot can't maintain "target lock". Each frame it sees "an enemy at (0.3, 0.5)" but doesn't know if it's the SAME enemy it was chasing.

```python
class ObjectTracker:
    """
    ByteTrack-style tracker using IoU matching.

    For each frame:
    1. Predict where existing tracks should be (using velocity)
    2. Match new detections to tracks using IoU
    3. Update matched tracks
    4. Create new tracks for unmatched detections
    5. Remove stale tracks (not seen for N frames)
    """

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        # Returns objects with persistent IDs
        # TrackedObject has: id, x, y, vx, vy, class_name, confidence, age
```

### 2. Strategist (Goal Selection)

**Architecture:** Transformer Encoder (4 layers, 256 dim, 4 heads)

**Input:** State summaries from last 60 seconds (1 per second = 60 tokens)

**Output:**
- Goal embedding: 32-64 dim continuous vector
- Mode logits: 5-class (FIGHT, LOOT, FLEE, EXPLORE, CAUTIOUS)

```python
class Strategist(nn.Module):
    def __init__(self, state_dim=128, hidden_dim=256, goal_dim=64, num_modes=5):
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=120)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=4
        )
        self.goal_head = nn.Linear(hidden_dim, goal_dim)
        self.mode_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, state_history):  # [B, T=60, state_dim]
        x = self.state_encoder(state_history)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        current = x[:, -1]  # Last timestep

        goal = self.goal_head(current)
        mode_logits = self.mode_head(current)
        return goal, mode_logits
```

### 3. Tactician (Target Selection)

**Architecture:** Cross-Attention with object embeddings

**Input:**
- Objects: Variable-size set of tracked objects
- Goal: 64-dim embedding from Strategist
- Recent states: Last 10-20 frames

**Output:**
- Target: Attention weights over objects (soft selection)
- Approach: 4-dim vector [vx, vy, urgency, aggression]

```python
class Tactician(nn.Module):
    def __init__(self, object_dim=32, goal_dim=64, hidden_dim=128):
        self.object_encoder = nn.Linear(object_dim, hidden_dim)
        self.goal_encoder = nn.Linear(goal_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=2)
        self.approach_head = nn.Linear(hidden_dim, 4)

    def forward(self, objects, goal):
        # objects: [B, N_objects, object_dim]
        # goal: [B, goal_dim]

        obj_embed = self.object_encoder(objects)  # [B, N, hidden]
        goal_query = self.goal_encoder(goal).unsqueeze(1)  # [B, 1, hidden]

        # Cross-attention: goal attends to objects
        # This learns "given this goal, which object matters?"
        attn_out, attn_weights = self.cross_attn(
            query=goal_query.transpose(0, 1),
            key=obj_embed.transpose(0, 1),
            value=obj_embed.transpose(0, 1)
        )

        target_embed = attn_out.transpose(0, 1).squeeze(1)  # [B, hidden]
        approach = self.approach_head(target_embed)  # [B, 4]

        # attn_weights: [B, 1, N] - which object is targeted
        return attn_weights.squeeze(1), approach
```

### 4. Executor (Frame-by-Frame Control)

**Architecture:** Mamba (State Space Model)

**Why Mamba over LSTM?**
- O(1) per-step inference (LSTM needs to process full history)
- Better at long sequences
- Hardware-efficient on modern GPUs
- 2024 state-of-the-art for sequence modeling

```python
# Using mamba-ssm library
from mamba_ssm import Mamba

class Executor(nn.Module):
    def __init__(self, state_dim=64, goal_dim=64, target_dim=32, hidden_dim=256):
        self.input_proj = nn.Linear(state_dim + goal_dim + target_dim, hidden_dim)

        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=64,
            d_conv=4,
            expand=2
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [mouse_x, mouse_y, click_logit, ability]
        )

    def forward(self, state, goal, target_info, hidden=None):
        x = torch.cat([state, goal, target_info], dim=-1)
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dim

        x = self.mamba(x)
        x = x.squeeze(1)

        action = self.action_head(x)
        return action
```

---

## State Representations

### Tracked Object Features (per object)

```python
object_features = [
    # Identity (6 dims)
    track_id,           # Persistent ID (embedded)
    class_id,           # Object type (embedded)
    confidence,         # Detection confidence
    age,                # Frames since first seen
    frames_visible,     # How many frames visible in last N
    frames_invisible,   # Frames since last seen

    # Position (4 dims)
    x, y,               # Normalized position (0-1)
    distance_to_player, # Distance from player
    angle_to_player,    # Angle from player (radians)

    # Velocity (4 dims)
    vx, vy,             # Velocity estimate
    speed,              # Magnitude
    heading,            # Direction of movement

    # Combat (4 dims, enemies only)
    threat_level,       # Estimated danger
    hp_estimate,        # If visible
    is_attacking_us,    # Boolean
    time_since_damage,  # If this enemy damaged us
]
# Total: ~18-20 dims per object
```

### Player State Features

```python
player_features = [
    # Position (4 dims)
    x, y,               # Screen position
    vx, vy,             # Velocity

    # Health (4 dims)
    hp,                 # Current health (0-1)
    hp_delta,           # Recent HP change
    shield,             # Current shield (0-1)
    shield_delta,       # Recent shield change

    # Combat (4 dims)
    is_attacking,       # Currently attacking
    time_attacking,     # Duration of current combat
    last_damage_taken,  # Time since damaged
    danger_level,       # Aggregate nearby threats

    # Context (4 dims)
    idle_time,          # Time without targets
    exploration_dir_x,  # Current exploration direction
    exploration_dir_y,
    near_edge,          # Close to map boundary
]
# Total: 16 dims
```

### Goal Embedding (from Strategist)

```python
goal_embedding = [
    # 64-dimensional learned embedding that captures:
    # - What mode we're in (fight/loot/flee/explore)
    # - How aggressive/defensive
    # - Time pressure
    # - Risk tolerance
    # - Target priority preferences
    #
    # This is LEARNED, not hand-designed.
    # The embedding space develops semantics through training.
]
```

---

## Training Strategy

### Phase 1: Behavior Cloning from Human Recordings

Train each component to match human behavior:

```
Human recordings (V1 data!)
       │
       ├──▶ Strategist: Predict mode from 60s context
       │
       ├──▶ Tactician: Predict target from mode + state
       │
       └──▶ Executor: Predict actions from target + goal
```

**Key:** We can reuse your existing recorded gameplay data!

### Phase 2: VLM Corrections (Self-Improvement)

VLM corrections are now RICHER:

```json
{
    "strategist_correction": {
        "current_mode": "FIGHT",
        "should_be": "FLEE",
        "reason": "HP critically low, enemies too strong"
    },
    "tactician_correction": {
        "current_target": "enemy_003",
        "should_target": "box_007",
        "reason": "Health pack nearby, prioritize survival"
    },
    "executor_correction": {
        "current_action": {"x": 0.5, "y": 0.5, "click": false},
        "should_be": {"x": 0.7, "y": 0.3, "click": true},
        "reason": "Moving too slowly toward target"
    }
}
```

### Phase 3: Optional Reinforcement Learning

If game environment available:

```python
reward = (
    + damage_dealt * 1.0
    + loot_collected * 0.5
    - damage_taken * 2.0
    - death_penalty * 10.0
)
```

---

## File Structure

```
darkorbit_bot/
├── reasoning/           # V1 (unchanged)
│   ├── policy_network.py
│   ├── state_builder.py
│   ├── bot_controller.py
│   └── ...
│
├── v2/                  # V2 (new)
│   ├── __init__.py
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── tracker.py          # ByteTrack implementation
│   │   └── state_encoder.py    # Rich state representation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── strategist.py       # Transformer for goal selection
│   │   ├── tactician.py        # Cross-attention for targeting
│   │   ├── executor.py         # Mamba for motor control
│   │   └── unified.py          # Combined hierarchical model
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_executor.py   # Train motor control
│   │   ├── train_tactician.py  # Train target selection
│   │   ├── train_strategist.py # Train mode selection
│   │   └── train_full.py       # End-to-end training
│   │
│   ├── bot_controller_v2.py    # New main controller
│   └── config.py               # V2 configuration
│
├── data/
│   ├── recordings/      # Shared with V1
│   └── checkpoints/
│       ├── v1/          # V1 models
│       └── v2/          # V2 models
│           ├── strategist.pt
│           ├── tactician.pt
│           └── executor.pt
```

---

## Migration Path

### Step 1: Add Object Tracking (works with V1)

Add ByteTrack to current pipeline. This improves V1 AND prepares for V2.

### Step 2: Build Executor First

Train Mamba executor on human data. This gives a working (basic) V2 bot.

### Step 3: Add Tactician

Add target selection. Now V2 can properly track and engage.

### Step 4: Add Strategist

Add mode selection. Now V2 has full hierarchical reasoning.

### Step 5: VLM Integration

Connect VLM corrections to all three layers.

---

## Comparison: V1 vs V2

| Aspect | V1 (Current) | V2 (New) |
|--------|--------------|----------|
| **Architecture** | Bi-LSTM | Hierarchical (Transformer + Mamba) |
| **Temporal** | ~1.7s context | 60s+ context |
| **Modes** | Binary (PASSIVE/AGGRESSIVE) | 5+ continuous goals |
| **Target tracking** | None (per-frame) | Persistent IDs |
| **Inference** | ~5ms | ~16ms (still fast!) |
| **Target selection** | Implicit in action | Explicit attention |
| **Explainability** | Black box | Can inspect goals + targets |
| **State size** | 128 dim | ~200+ dim with objects |

---

## Dependencies

```
# New dependencies for V2
mamba-ssm>=1.2.0        # State space model
# OR if mamba not available:
# s4>=0.1.0             # Alternative SSM

# Existing (already have)
torch>=2.0
numpy
ultralytics            # YOLO
```

---

## Quick Start (after implementation)

```bash
# Train V2 on existing recordings
python -m darkorbit_bot.v2.training.train_full --data data/recordings --epochs 20

# Run V2 bot
python -m darkorbit_bot.v2.bot_controller_v2 --monitor 1

# Run V1 bot (unchanged)
python -m darkorbit_bot.reasoning.bot_controller --monitor 1
```

---

## Summary

V2 is a modern hierarchical architecture that:

1. **Separates concerns** - Strategy, tactics, and execution at different timescales
2. **Tracks objects** - Persistent IDs for target lock
3. **Uses modern models** - Transformer for strategy, Mamba for execution
4. **Flexible goals** - Continuous embeddings instead of binary modes
5. **Explicit targeting** - Can inspect and override target selection
6. **Reuses V1 data** - Your recordings still work!
7. **Coexists with V1** - Both systems can run independently

Total latency: ~16ms (well under 50ms target)
