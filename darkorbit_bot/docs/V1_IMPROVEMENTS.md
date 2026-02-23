# V1 Bot Improvement Roadmap

This document outlines potential improvements for the V1 Bi-LSTM architecture before transitioning to V2.

---

## High Impact Improvements

### 1. Better Target Prioritization

**Problem**: Bot picks nearest enemy, ignoring threat level, health, and reward value.

**Solution**:
- Add target scoring system: `score = threat_weight * threat + reward_weight * reward - distance_weight * distance`
- Learn priority from recordings (which enemies you attack first)
- Add features to state: `enemy_type_one_hot`, `estimated_threat`, `estimated_reward`

**Implementation**:
```python
# In state_builder.py - add to _detection_to_features()
def _calculate_target_priority(self, det):
    THREAT_MAP = {
        'Sibelon': 0.9, 'Mordon': 0.7, 'Struener': 0.6,
        'Saimon': 0.5, 'Lordakia': 0.3, 'Devo': 0.2
    }
    REWARD_MAP = {
        'Sibelon': 0.9, 'Mordon': 0.7, 'Struener': 0.5,
        'Saimon': 0.4, 'Lordakia': 0.2, 'Devo': 0.1
    }
    return THREAT_MAP.get(det.class_name, 0.5), REWARD_MAP.get(det.class_name, 0.5)
```

**VLM Enhancement**:
- Add to prompt: "Which enemy should be prioritized and why?"
- VLM can suggest priority based on game knowledge
- Save VLM priority decisions as training data

---

### 2. Combat Timing/Rhythm Learning

**Problem**: Bot spams abilities without understanding cooldowns or optimal timing.

**Solution**:
- Track time since last ability use in state
- Learn rocket timing from recordings (when player fires rockets)
- Add cooldown awareness features

**Implementation**:
```python
# New features for player state
combat_timing_features = [
    time_since_last_rocket,      # 0-1 normalized (0=just fired, 1=ready)
    time_since_attack_toggle,    # How long attacking
    enemy_health_estimate,       # If detectable from size/animation
    combat_duration,             # Time in current engagement
]
```

**VLM Enhancement**:
- Add to prompt: "Was the rocket timing good? Should bot have waited for cooldown?"
- Track VLM feedback on ability usage timing
- Generate "optimal timing" corrections

---

### 3. Escape/Survival Behavior

**Problem**: Bot has no retreat logic when low health or outnumbered.

**Solution**:
- Add health threshold for retreat mode
- Learn escape patterns from death-avoidance recordings
- Detect "danger zones" (multiple enemies)

**Implementation**:
```python
# In bot_controller.py
def should_retreat(self, health, num_enemies, nearest_enemy_dist):
    # Learned thresholds from recordings
    if health < 0.3 and num_enemies > 2:
        return True
    if health < 0.15:
        return True
    return False

# Add RETREAT mode to policy
# Retreat head outputs: escape_direction, should_use_jump, should_cloak
```

**VLM Enhancement**:
- Add to prompt: "Is the bot in danger? Should it retreat?"
- Detect near-death situations and flag for learning
- "What should the bot have done to survive?"

---

### 4. Multi-Target Awareness

**Problem**: Bot loses track when current target dies, slow to switch.

**Solution**:
- Track top-3 targets instead of just nearest
- Pre-compute "next target" before current dies
- Add target persistence features

**Implementation**:
```python
# In state_builder.py
def build_multi_target_features(self, detections, current_target):
    targets = self._prioritize_detections(detections)[:3]
    features = []
    for i, target in enumerate(targets):
        features.extend([
            target.x_center, target.y_center,
            target.confidence,
            1.0 if target == current_target else 0.0  # is_current
        ])
    # Pad if fewer than 3 targets
    while len(features) < 12:
        features.append(0.0)
    return features
```

**VLM Enhancement**:
- Ask VLM: "What should be the next target after current one dies?"
- Evaluate target switching speed: "Did bot switch targets efficiently?"

---

## Medium Impact Improvements

### 5. VLM Feedback Loop (Self-Training)

**Problem**: VLM critiques are logged but not used to improve model.

**Solution**:
- Weight training samples by VLM quality scores
- Generate synthetic "good" actions from VLM corrections
- Auto-delete recordings VLM rates as "bad"

**Implementation**:
```python
# In train.py - weight samples by VLM score
def load_with_vlm_weights(data_dir):
    for sequence_file in data_dir.glob("*.json"):
        data = json.load(sequence_file)

        # Check for VLM annotations
        vlm_score = data.get('vlm_quality_score', 1.0)

        # Skip bad samples
        if vlm_score < 0.3:
            continue

        # Weight good samples higher
        sample_weight = vlm_score ** 2  # Square to emphasize good samples

        yield data, sample_weight
```

**VLM Enhancement**:
- Output confidence score (0-1) for each critique
- Generate corrected action sequences, not just single frames
- "Replay" bad sequences with VLM-suggested actions

---

### 6. Movement Pattern Refinement

**Problem**: Orbiting radius not optimized per enemy type.

**Solution**:
- Learn optimal orbit distance per enemy from recordings
- Detect map obstacles (asteroids, edges)
- Add "comfort zone" features

**Implementation**:
```python
# Enemy-specific orbit distances (learned from recordings)
OPTIMAL_ORBIT = {
    'Sibelon': 0.15,   # Stay far - high damage
    'Mordon': 0.12,
    'Struener': 0.10,
    'Saimon': 0.08,
    'Lordakia': 0.06,  # Can get close - weak
    'Devo': 0.05,
}

# Add to state: distance_to_optimal_orbit
def orbit_quality(self, current_dist, enemy_type):
    optimal = OPTIMAL_ORBIT.get(enemy_type, 0.1)
    return 1.0 - min(1.0, abs(current_dist - optimal) / optimal)
```

**VLM Enhancement**:
- Ask: "Is the orbit distance appropriate for this enemy?"
- "Is the bot too close/far from the enemy?"
- Learn enemy-specific distances from VLM feedback

---

### 7. State History Compression

**Problem**: 50-frame sequences contain mostly redundant info.

**Solution**:
- Encode "events" as discrete features instead of raw frames
- Compress history into: last_kill_time, last_damage_time, last_loot_time
- Focus sequence on "important moments"

**Implementation**:
```python
# Event-based state compression
class EventEncoder:
    def __init__(self):
        self.events = []  # (time, event_type, data)

    def add_event(self, event_type, data=None):
        self.events.append((time.time(), event_type, data))
        # Keep last 20 events
        self.events = self.events[-20:]

    def get_event_features(self):
        now = time.time()
        features = {
            'time_since_kill': 999,
            'time_since_damage': 999,
            'time_since_loot': 999,
            'kills_last_30s': 0,
            'damage_events_last_30s': 0,
        }
        for t, event_type, data in self.events:
            age = now - t
            if event_type == 'kill':
                features['time_since_kill'] = min(features['time_since_kill'], age)
                if age < 30:
                    features['kills_last_30s'] += 1
            # ... etc
        return list(features.values())
```

**VLM Enhancement**:
- VLM already detects events (kill, damage) - feed back to EventEncoder
- Use VLM to validate event detection accuracy

---

### 8. Confidence Calibration

**Problem**: Bot acts with same confidence on easy and hard decisions.

**Solution**:
- Add uncertainty output to network
- Fall back to safe behavior when uncertain
- Ask VLM for help on low-confidence frames

**Implementation**:
```python
# In policy_network.py - add confidence head
self.confidence_head = nn.Sequential(
    nn.Linear(hidden_size * 2, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

def get_action_with_confidence(self, state_seq):
    action = self.get_action(state_seq)
    confidence = self.confidence_head(context).item()

    if confidence < 0.3:
        # Fall back to safe behavior
        action['should_fire'] = False
        action['use_vlm'] = True  # Request VLM guidance

    return action, confidence
```

**VLM Enhancement**:
- Only query VLM on low-confidence frames (saves compute)
- VLM provides "ground truth" for confidence calibration
- Train confidence head to match VLM agreement

---

## Lower Impact (Nice to Have)

### 9. Map Awareness

**Problem**: Bot doesn't know safe vs dangerous zones.

**Solution**:
- Build map from screenshots over time
- Mark areas where deaths occurred
- Navigate toward farming spots

**VLM Enhancement**:
- Ask VLM: "What area of the map is this?"
- "Is this a safe zone or dangerous zone?"
- Build map annotations from VLM responses

---

### 10. Enemy Type Specialization

**Problem**: Same combat pattern for all enemies.

**Solution**:
- Condition behavior on enemy type
- Different orbit patterns per enemy
- Learn enemy attack patterns

**VLM Enhancement**:
- "What is the optimal tactic against Sibelon?"
- Build enemy-specific tactic database from VLM

---

### 11. Loot Efficiency

**Problem**: Suboptimal collection routes.

**Solution**:
- Plan collection path for multiple boxes
- Skip low-value loot during combat
- Prioritize cargo boxes over bonus boxes

**VLM Enhancement**:
- "Which boxes should be collected first?"
- "Should bot stop fighting to collect this box?"

---

### 12. Session Memory

**Problem**: No adaptation within session.

**Solution**:
- Track what worked (which tactics led to kills)
- Adapt to current map spawn patterns
- Remember dangerous areas from this session

**Implementation**:
```python
class SessionMemory:
    def __init__(self):
        self.successful_tactics = defaultdict(int)
        self.failed_tactics = defaultdict(int)
        self.danger_zones = []  # Areas where we took damage

    def record_outcome(self, tactic, enemy_type, success):
        key = (tactic, enemy_type)
        if success:
            self.successful_tactics[key] += 1
        else:
            self.failed_tactics[key] += 1

    def get_recommended_tactic(self, enemy_type):
        # Return tactic with best success rate this session
        pass
```

---

## VLM System Improvements

### Current VLM Capabilities
- Screenshot analysis with bounding boxes
- Action critique (good/bad rating)
- Suggested corrections
- Combat tactic detection
- **NEW: System prompt with game knowledge** (implemented!)

### System Prompt (Implemented)

The VLM now receives a system prompt with:

1. **Enemy Knowledge**
   - Threat levels: Sibelon > Mordon > Struener > Saimon > Lordakia > Devo
   - Optimal distances per enemy type (5-20% screen)
   - Recommended tactics per enemy

2. **Combat Tactics Ranking**
   - Orbiting (best) > Kiting > Hit-and-run > Face-tanking (bad) > No pattern (worst)

3. **Control Mappings**
   - Ctrl = attack toggle, Space = rocket, Shift = special

4. **Good/Bad Behavior Lists**
   - What good bots do (fast attack, proper distance, smooth targeting)
   - What bad bots do (delayed attack, face-tanking, erratic movement)

5. **False Positive Detection**
   - How to verify YOLO bounding boxes are real targets

### Proposed VLM Enhancements

#### 1. Sequence Analysis Mode
```python
# Instead of single frames, analyze 5-10 frame sequences
SEQUENCE_PROMPT = """
Analyze this sequence of {n} frames spanning {duration} seconds.

Frame-by-frame timeline:
{timeline}

Questions:
1. Did the bot's actions show a coherent strategy?
2. What pattern was the bot attempting? (orbiting/kiting/etc)
3. Was the pattern executed well?
4. What should the bot do NEXT based on this sequence?
"""
```

#### 2. Comparative Learning
```python
# Show VLM two sequences - one good, one bad
COMPARE_PROMPT = """
Compare these two combat sequences:

SEQUENCE A (labeled as GOOD):
{good_sequence}

SEQUENCE B (labeled as BAD):
{bad_sequence}

What makes A better than B?
What specific actions differentiate good from bad play?
"""
```

#### 3. Strategy Explanation
```python
# Have VLM explain optimal strategy
STRATEGY_PROMPT = """
You see a {enemy_type} enemy at distance {distance}.
Player health: {health}
Nearby obstacles: {obstacles}

Explain the optimal combat strategy:
1. What tactic should be used?
2. What distance should be maintained?
3. When should abilities be used?
4. What are the danger signs to watch for?
"""
```

#### 4. Training Data Generation
```python
# VLM generates synthetic training examples
GENERATE_PROMPT = """
Given this game state:
{state_description}

Generate 5 example "expert" actions with explanations:
1. (x, y, click, ctrl, space) - explanation
2. ...

These will be used to train the bot.
"""
```

#### 5. Error Analysis Mode
```python
# Detailed analysis of mistakes
ERROR_PROMPT = """
The bot died in this sequence. Analyze what went wrong:

{death_sequence}

1. When did things start going wrong?
2. What was the first mistake?
3. What could have prevented the death?
4. Rate the severity of each mistake (1-10)
"""
```

---

## Implementation Priority

### Phase 1 (Quick Wins)
1. ✅ Movement pattern features (already added)
2. ✅ Keyboard action learning (already added)
3. ✅ Mode selector (already added)
4. VLM feedback loop integration

### Phase 2 (Medium Effort)
5. Target prioritization with threat/reward
6. Combat timing features
7. Multi-target awareness
8. Confidence calibration

### Phase 3 (Before V2)
9. Escape/survival behavior
10. State history compression
11. Enemy type specialization

### Phase 4 (Optional)
12. Map awareness
13. Session memory
14. Loot efficiency

---

## Migration to V2

V2 will handle most of these improvements architecturally:

| V1 Improvement | V2 Equivalent |
|----------------|---------------|
| Target prioritization | Tactician with attention |
| Combat timing | Executor with SSM memory |
| Multi-target | ByteTrack persistence |
| Escape behavior | Strategist mode selection |
| State compression | Hierarchical state encoding |
| Confidence | Per-level confidence outputs |

V1 improvements that transfer to V2:
- VLM feedback system (same prompts work)
- Movement profiles (humanization layer)
- YOLO detection model (shared)
- Recording format (compatible)

---

## Notes

- All improvements are backwards compatible with existing recordings
- New features can be disabled for old models
- VLM improvements are independent of model architecture
- Focus on VLM feedback loop for quickest gains without retraining
