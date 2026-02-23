# DarkOrbit V2 Feature Specification

Complete feature map for the V2 hierarchical bot architecture.

---

## Feature Priority Legend

- 🔴 **Critical** — Bot won't work well without this
- 🟡 **Important** — Significantly improves performance
- 🟢 **Nice-to-have** — Adds nuance but not essential

---

## 1. Shared Features (All 3 Models)

| Feature | Type | Range | Why Needed |
|---------|------|-------|------------|
| 🔴 Player X, Y | float | 0-1 | Core position awareness |
| 🔴 Player HP % | float | 0-1 | Survival decisions |
| 🔴 Player Shield % | float | 0-1 | Damage buffer |
| 🔴 Is Attacking | bool | 0/1 | Combat state |
| 🟡 Current Speed | float | 0-1 | Movement detection |

---

## 2. Strategist-Specific (Long-term Decisions)

### Temporal Trends (60-second history)

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 HP Trend (last 60s) | float | Damage rate → fight/flee |
| 🔴 Shield Trend | float | Regen vs drain rate |
| 🔴 Kill Count | int | Are we winning? |
| 🔴 Death/Near-Death Events | int | Should we be more cautious? |
| 🟡 Loot Collected | int | Productivity measure |
| 🟡 Time in Combat | float | Engagement duration |
| 🟡 Time Idle | float | Should we explore? |

### Resource State

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Ammo % | float | Can we fight? |
| 🔴 Rockets Available | bool | Heavy firepower ready? |
| 🟡 Special Ammo Active | bool | MCB-50, etc. |
| 🟡 Credit Box Count | int | Worth staying here? |

### Map Awareness

| Feature | Type | Why Needed |
|---------|------|------------|
| 🟡 Near Portal | bool | Escape route available |
| 🟡 Near Base | bool | Safe zone nearby |
| 🟡 Map Zone Risk Level | float | PvP danger (estimated) |
| 🟢 Time in Current Map | float | Should we switch? |

---

## 3. Tactician-Specific (Target Selection)

### Per-Object Features

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Object Class | enum | Enemy, Loot, Player, Portal |
| 🔴 Distance to Player | float | Reachability |
| 🔴 Angle to Player | float | Which direction |
| 🔴 Object Velocity | vec2 | Is it approaching/fleeing? |
| 🔴 Is Attacking Us | bool | Threat priority |
| 🟡 Track Age | int | How long we've seen it |
| 🟡 Track Confidence | float | Reliable detection? |

### Enemy-Specific

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Enemy Type | enum | NPC type affects behavior |
| 🔴 Enemy HP % | float | Can we kill it? |
| 🟡 Enemy Attacking Others | bool | Distracted = easy kill |
| 🟡 Enemy Speed | float | Can we catch it? |
| 🟡 Time Since Enemy Attacked | float | Aggro cooldown |

### Cooldown Tracking

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Our Cloak CD | float | 0 = ready, >0 = waiting |
| 🔴 Our EMP CD | float | Stun available? |
| 🔴 Our Insta-Shield CD | float | Emergency ready? |
| 🟡 Est. Enemy EMP CD | float | Safe to approach? |
| 🟡 Est. Enemy Cloak CD | float | Will they escape? |
| 🟢 Our Drone Formation | enum | Offensive/Defensive? |

### Win Probability Calculator

| Feature | Type | Why Needed |
|---------|------|------------|
| 🟡 Our DPS Estimate | float | Based on config + ammo |
| 🟡 Enemy DPS Estimate | float | Based on ship type |
| 🟡 Time to Kill Enemy | float | Enemy HP / our DPS |
| 🟡 Time Enemy Kills Us | float | Our HP / their DPS |
| 🔴 Win Probability | float | TTK comparison |

---

## 4. Executor-Specific (Precise Actions)

### Motor Control Inputs

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Target Screen X, Y | float | Where to move mouse |
| 🔴 Current Mouse X, Y | float | Current position |
| 🔴 Mouse Velocity | vec2 | Smooth movement |
| 🔴 Goal Embedding | vec64 | Strategy context |
| 🔴 Target Info | vec32 | Tactician output |

### Action Context

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Should Click | bool | Fire weapon |
| 🔴 Click Type | enum | Left/Right |
| 🟡 Hotkey to Press | enum | 1-9, Q, E, R, Space, Ctrl |
| 🟡 Time Since Last Click | float | Rate limiting |
| 🟡 Time Since Last Hotkey | float | Cooldown respect |

### Urgency Signals

| Feature | Type | Why Needed |
|---------|------|------------|
| 🔴 Urgency | float | Movement speed factor |
| 🔴 Aggression | float | Click frequency |
| 🟡 Precision Required | float | Careful aim vs spam |

---

## 5. Feature Distribution Matrix

| Feature Category | Strategist | Tactician | Executor |
|------------------|:----------:|:---------:|:--------:|
| Player Position | Summary | ✅ | ✅ |
| Player HP/Shield | ✅ Trend | ✅ | via Goal |
| Object Positions | Count | ✅ Full | Target only |
| Object Velocities | ❌ | ✅ | Target only |
| Enemy HP | ❌ | ✅ | ❌ |
| Cooldowns (Ours) | ❌ | ✅ | ❌ |
| Cooldowns (Enemy) | ❌ | ✅ | ❌ |
| Win Probability | ❌ | ✅ | ❌ |
| Mouse X,Y | ❌ | ❌ | ✅ |
| Hotkeys | ❌ | ❌ | ✅ |
| Goal Embedding | Produces → | Uses → | Uses → |
| Target Info | ❌ | Produces → | Uses → |

---

## 6. Current Implementation Status

| Feature | Status | How to Add |
|---------|--------|------------|
| Player HP/Shield | ✅ Implemented | — |
| Object Tracking | ✅ Implemented | ByteTrack |
| Enemy HP | ⚠️ Not tracked | OCR or YOLO HP detection |
| Cooldowns | ❌ Missing | Track keypresses + timer |
| Win Probability | ❌ Missing | Compute from HP + DPS |
| Ammo % | ⚠️ Not tracked | OCR ammo counter |
| Map Zone | ⚠️ Not tracked | OCR minimap |

---

## 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────┤
│  Screen Capture → YOLO → ByteTrack → TrackedObjects            │
│  OCR (future) → HP%, Ammo%, Cooldowns                          │
│  Keylogger → Cooldown Timer Start                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATE ENCODER V2                             │
├─────────────────────────────────────────────────────────────────┤
│  Combines all sources into:                                     │
│  • Player Features (16 dim)                                     │
│  • Object Features (20 dim × N objects)                         │
│  • Context Features (16 dim)                                    │
│  • Temporal Summaries (for Strategist)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  STRATEGIST  │    │  TACTICIAN   │    │   EXECUTOR   │
│   (1 Hz)     │    │   (10 Hz)    │    │   (60 Hz)    │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ Sees:        │    │ Sees:        │    │ Sees:        │
│ • 60s trends │───▶│ • Objects    │───▶│ • Target pos │
│ • HP/Shield  │    │ • Goal embed │    │ • Goal embed │
│ • Kill count │    │ • Cooldowns  │    │ • Target info│
│              │    │ • Win prob   │    │ • Mouse pos  │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ Outputs:     │    │ Outputs:     │    │ Outputs:     │
│ • Goal embed │    │ • Target ID  │    │ • Mouse X,Y  │
│ • Mode       │    │ • Target info│    │ • Click type │
│              │    │ • Approach   │    │ • Hotkey     │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 8. Priority Implementation Order

1. **Phase 1: Core (Current)**
   - ✅ Player position, HP, Shield
   - ✅ Object tracking (ByteTrack)
   - ✅ Basic state encoding

2. **Phase 2: Tactical Intelligence**
   - ⬜ Enemy HP detection
   - ⬜ Cooldown tracking
   - ⬜ Win probability calculation

3. **Phase 3: Strategic Awareness**
   - ⬜ Ammo tracking
   - ⬜ Map zone detection
   - ⬜ Portal/base proximity

4. **Phase 4: Advanced**
   - ⬜ Enemy cooldown estimation
   - ⬜ Multi-enemy threat assessment
   - ⬜ Predictive movement
