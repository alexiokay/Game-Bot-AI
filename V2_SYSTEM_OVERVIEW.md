# V2 System Overview - Complete Picture

This document provides a comprehensive, visualization-ready overview of the entire V2 DarkOrbit bot system.

---

## 1. The Big Picture

```mermaid
flowchart TB
    subgraph DATA["📊 DATA COLLECTION"]
        A[Human Gameplay] --> B[FilteredRecorder]
        B --> C[Recordings JSON]
        C --> D[VLM Labeler]
        D --> E[Labeled Dataset]
    end

    subgraph TRAINING["🧠 TRAINING PIPELINE"]
        E --> F[Train Executor]
        E --> G[Train Tactician]
        E --> H[Train Strategist]
    end

    subgraph INFERENCE["🎮 LIVE BOT"]
        I[Screen Capture] --> J[YOLO]
        J --> K[ByteTrack]
        K --> L[State Encoder]
        L --> M[Strategist]
        L --> N[Tactician]
        L --> O[Executor]
        M --> N
        N --> O
        O --> P[Mouse/Keyboard]
    end

    TRAINING --> INFERENCE
```

---

## 2. The Hierarchical Brain

The V2 bot thinks at **three different speeds**:

| Layer | Model | Speed | Responsibility |
|-------|-------|-------|----------------|
| **Strategist** | Transformer | 1 Hz | "What should I be doing?" (FIGHT / FLEE / LOOT) |
| **Tactician** | Cross-Attention | 10 Hz | "Which target?" (Select from detected objects) |
| **Executor** | Mamba/LSTM | 60 Hz | "Exactly where to move mouse, when to click" |

```mermaid
flowchart LR
    subgraph S["STRATEGIST (1Hz)"]
        S1[60s State History] --> S2[Transformer]
        S2 --> S3[Goal Embedding]
        S2 --> S4[Mode: FIGHT/FLEE/LOOT]
    end

    subgraph T["TACTICIAN (10Hz)"]
        T1[Object List] --> T2[Cross-Attention]
        S3 --> T2
        T2 --> T3[Target Selection]
        T2 --> T4[Approach Vector]
    end

    subgraph E["EXECUTOR (60Hz)"]
        E1[Current State] --> E2[Mamba]
        S3 --> E2
        T3 --> E2
        T4 --> E2
        E2 --> E3[Mouse X,Y]
        E2 --> E4[Click Type]
        E2 --> E5[Hotkey]
    end

    S --> T --> E
```

---

## 3. Perception Pipeline

Before any decision-making, we need to **see** the game.

```mermaid
flowchart LR
    A[Screenshot 60fps] --> B[YOLO Detection]
    B --> C[ByteTrack]
    C --> D[TrackedObject List]
    D --> E[StateEncoderV2]
    E --> F[Neural Network Features]

    subgraph ByteTrack
        C1[Match Detections to Tracks]
        C2[Assign Persistent IDs]
        C3[Estimate Velocities]
    end
```

**Key Concept:** ByteTrack gives each object a **persistent ID**.
- Frame 1: Enemy at (0.3, 0.5) → ID: `enemy_001`
- Frame 2: Enemy moves to (0.4, 0.5) → Still `enemy_001`
- Now the bot can "lock on" and track the same enemy across frames.

---

## 4. The Training Pipeline

### 4.1 Data Flow

```mermaid
flowchart TB
    subgraph RECORD["Phase 1: Record"]
        R1[Play Game] --> R2[FilteredRecorder]
        R2 --> R3[sequence_XXXX.json]
        R2 --> R4[Screenshots]
    end

    subgraph LABEL["Phase 2: VLM Labeling"]
        R3 --> L1[VLM Labeler]
        R4 --> L1
        L1 --> L2["ground_truth.json
        - Exact Health %
        - Tactic Label
        - Intent"]
    end

    subgraph TRAIN["Phase 3: Component Training"]
        L2 --> T1[Executor Training]
        L2 --> T2[Tactician Training]
        L2 --> T3[Strategist Training]
    end
```

### 4.2 The Causality Bridge Problem

> "How does a 2-second VLM window understand 10-second consequences?"

**Answer:** It doesn't need to — that's the Strategist's job.

```mermaid
sequenceDiagram
    participant VLM as VLM Labeler
    participant Data as Training Data
    participant Strat as Strategist

    Note over VLM: Sees T=60s screenshot
    VLM->>Data: Labels "FLEE" (player running)
    
    Note over Strat: Sees T=0s to T=60s history
    Strat->>Data: Sees damage at T=50s
    
    Note over Strat: Learns correlation
    Strat-->>Strat: "Big damage → FLEE later"
```

---

## 5. Executor Action Space (Upgraded)

The Executor outputs:

| Output | Type | Values |
|--------|------|--------|
| `mouse_x` | Continuous | 0.0 - 1.0 |
| `mouse_y` | Continuous | 0.0 - 1.0 |
| `click_type` | Discrete (Softmax) | None, Left, Right |
| `hotkey` | Discrete (Softmax) | None, 1-9, Q, E, R, Space, Ctrl... |

This allows the bot to use any ability, not just click.

---

## 6. VLM "Sliding Window" Analysis

Instead of single screenshots, we send **frame sequences** to the VLM.

```mermaid
flowchart LR
    subgraph Window["2-Second Window"]
        F1[Frame 1] 
        F2[Frame 2]
        F3[Frame 3]
        F4[Frame 4]
    end

    Window --> VLM[Gemini Flash]
    VLM --> JSON["{ health: 85, tactic: 'Kiting', intent: 'FIGHT' }"]
```

**Why?** A single frame can't show motion. With multiple frames, VLM can see:
- "Player is circling enemy" → Kiting
- "Player is running toward portal" → Fleeing
- "Player is stationary near boxes" → Looting

---

## 7. Complete System Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           V2 BOT RUNTIME                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│   │   YOLO      │───▶│  ByteTrack  │───▶│   State     │                      │
│   │  (10ms)     │    │   (2ms)     │    │  Encoder    │                      │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                      │
│                                                 │                             │
│         ┌───────────────────────────────────────┼─────────────────────┐      │
│         │                                       │                      │      │
│         ▼                                       ▼                      ▼      │
│   ┌───────────┐                          ┌───────────┐          ┌───────────┐│
│   │STRATEGIST │                          │ TACTICIAN │          │ EXECUTOR  ││
│   │  (1 Hz)   │─────Goal Embedding──────▶│  (10 Hz)  │─Target──▶│  (60 Hz)  ││
│   │Transformer│                          │Cross-Attn │          │   Mamba   ││
│   └───────────┘                          └───────────┘          └─────┬─────┘│
│         │                                                             │      │
│         │ Mode: FIGHT/FLEE/LOOT                                       │      │
│         │                                              ┌──────────────┘      │
│         ▼                                              ▼                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    HUMANIZED ACTION EXECUTOR                         │   │
│   │                    Bezier curves, variable timing                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│                            [ Mouse + Keyboard Output ]                       │
│                                                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                           VLM CRITIQUE (Async, 3s)                           │
│                    "You should have fled, not fought"                        │
│                    Corrections → Retrain All Layers                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. File Map

```
f:\dev\bot\
├── ARCHITECTURE_V2.md          # Technical architecture details
├── TRAINING_WORKFLOW_V2.md     # How to train the system
├── IMPLEMENTATION_PLAN_V2.md   # What needs to be built
├── V2_SYSTEM_OVERVIEW.md       # THIS FILE (Big Picture)
│
└── darkorbit_bot/
    └── v2/
        ├── perception/
        │   ├── tracker.py           # ByteTrack object tracking
        │   └── state_encoder.py     # Feature encoding
        │
        ├── models/
        │   ├── strategist.py        # Transformer (1Hz)
        │   ├── tactician.py         # Cross-Attention (10Hz)
        │   ├── executor.py          # Mamba/LSTM (60Hz)
        │   └── unified.py           # Combined model
        │
        ├── training/
        │   ├── train_executor.py
        │   ├── train_tactician.py
        │   ├── train_strategist.py
        │   └── vlm_labeler.py       # (To be built)
        │
        ├── bot_controller_v2.py     # Main runtime
        └── config.py                # All configuration
```

---

## 9. Summary Table

| Concept | Purpose | Key Tech |
|---------|---------|----------|
| **Hierarchical Temporal** | Different decisions at different speeds | Transformer + Mamba |
| **ByteTrack** | Persistent object IDs across frames | IoU matching |
| **VLM Coach** | Label tactics from video clips | Gemini 1.5 Flash |
| **Sliding Window** | Multi-frame context for VLM | 4-5 frames / 2s |
| **Causality Bridge** | Long-term learning from short-term labels | Strategist 60s history |
| **Complex Actions** | Full hotbar support | Softmax classification |

---

## 10. Next Steps (Implementation Order)

1. ✅ **Architecture Design** — Complete
2. ⬜ **Data Collection Upgrade** — Capture hotkeys in FilteredRecorder
3. ⬜ **Executor Upgrade** — Multi-head output (mouse + click + hotkey)
4. ⬜ **VLM Labeler** — Sliding window + tactical descriptions
5. ⬜ **Train Components** — Executor → Tactician → Strategist
6. ⬜ **Integration** — Connect all layers in bot_controller_v2.py
