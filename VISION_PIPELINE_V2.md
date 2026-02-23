# Vision Pipeline V2: Hybrid Detection System

Replacing a single overwhelmed YOLO model with specialized tools for each detection task.

---

## Problem Statement

### Current Approach (Single YOLO)

```
YOLO11 trying to detect EVERYTHING:
├── Enemies (6+ NPC types)
├── Enemy players
├── Ally ships
├── Loot boxes
├── Portals
├── Player ship
├── UI HP bar
├── UI Ammo counter
├── Cooldown icons
├── Minimap elements
├── Drones
├── Effects (EMP, Cloak, etc.)
└── ...50+ classes
```

**Problems:**
- 🔴 Too many classes → low accuracy
- 🔴 Confused by similar objects (drones vs ships)
- 🔴 YOLO can't read text/numbers
- 🔴 Small objects flicker in/out
- 🔴 Massive training data required

---

## Solution: Hybrid Vision Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID VISION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   YOLO11     │  │  COLOR SCAN  │  │   TIMERS     │          │
│  │  (Objects)   │  │ (Health Bars)│  │ (Cooldowns)  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         │          ┌──────────────┐          │                  │
│         │          │     OCR      │          │                  │
│         │          │  (Numbers)   │          │                  │
│         │          └──────┬───────┘          │                  │
│         │                 │                  │                  │
│         └────────────────┼──────────────────┘                  │
│                          ▼                                      │
│                 ┌─────────────────┐                            │
│                 │  UNIFIED STATE  │                            │
│                 │    BUILDER      │                            │
│                 └────────┬────────┘                            │
│                          │                                      │
│                          ▼                                      │
│                 ┌─────────────────┐                            │
│                 │  StateEncoderV2 │                            │
│                 └─────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```


---

## Detection Tasks Separation

### Task 1: Object Detection (YOLO11)

**Purpose:** Detect game entities in the world

**Classes (6 only):**
| Class | Description | Priority |
|-------|-------------|----------|
| `enemy_npc` | All NPC enemies (merged) | 🔴 Critical |
| `enemy_player` | Other players (PvP) | 🔴 Critical |
| `ally` | Friendly ships | 🟡 Medium |
| `loot` | All loot boxes | 🔴 Critical |
| `portal` | Jump portals | 🟡 Medium |
| `player_ship` | Our ship (reference) | 🟢 Low |

**Why merge NPC types?**
- Devo, Lordakia, Mordon → all just "enemy_npc"
- Type can be inferred from size/HP if needed later
- Fewer classes = higher accuracy

**Output:**
```python
[
    {"class": "enemy_npc", "x": 0.4, "y": 0.3, "confidence": 0.92},
    {"class": "loot", "x": 0.7, "y": 0.5, "confidence": 0.88},
]
```

---

### Task 2: Health Bar Reading (Color Scan)

**Purpose:** Read HP/Shield percentages from fixed UI regions

**Method:** Pixel color counting (no AI needed)

**UI Regions (1920x1080):**
| Region | Left | Top | Right | Bottom |
|--------|------|-----|-------|--------|
| HP Bar | 50 | 80 | 250 | 95 |
| Shield Bar | 50 | 100 | 250 | 115 |
| Target HP | 50 | 130 | 250 | 145 |

**Algorithm:**
```python
def read_hp_bar(screenshot, region):
    """
    Read HP percentage by counting colored pixels.
    
    HP bar: Green → Red gradient based on health
    """
    bar = screenshot.crop(region)
    bar_width = region[2] - region[0]
    
    # Count non-black pixels (the filled part)
    filled_pixels = 0
    for x in range(bar_width):
        pixel = bar.getpixel((x, bar.height // 2))
        if sum(pixel[:3]) > 50:  # Not black
            filled_pixels += 1
    
    return filled_pixels / bar_width
```

**Output:**
```python
{
    "player_hp": 0.85,
    "player_shield": 1.0,
    "target_hp": 0.42
}
```

---

### Task 3: Number Reading (OCR)

**Purpose:** Read exact numbers from UI (ammo, coordinates, credits, etc.)

**Method:** OCR using EasyOCR or Tesseract on fixed UI regions

**Libraries:**
| Library | Speed | Accuracy | Install |
|---------|-------|----------|---------|
| EasyOCR | ~50ms | High | `pip install easyocr` |
| Tesseract | ~30ms | Medium | Requires install |
| PaddleOCR | ~40ms | Very High | `pip install paddleocr` |

**UI Regions for OCR (1920x1080):**
| Region | Left | Top | Right | Bottom | Content |
|--------|------|-----|-------|--------|---------|
| Ammo Count | 1700 | 900 | 1850 | 930 | "12,500" |
| Rockets | 1700 | 935 | 1850 | 960 | "150" |
| Credits | 100 | 10 | 250 | 35 | "1,234,567" |
| Uridium | 260 | 10 | 400 | 35 | "50,000" |
| Coords X,Y | 1750 | 180 | 1900 | 210 | "24|18" |

**Algorithm:**
```python
import easyocr

class NumberReader:
    """OCR for reading numbers from fixed UI regions."""
    
    def __init__(self):
        # Initialize once (slow, ~2s)
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        # Precompile number-only pattern
        self.number_pattern = re.compile(r'[\d,\.]+')
    
    def read_region(self, screenshot, region) -> str:
        """Read text from a specific region."""
        crop = screenshot.crop(region)
        
        # Convert to numpy for EasyOCR
        img_np = np.array(crop)
        
        # OCR with number detection
        results = self.reader.readtext(img_np, detail=0)
        
        if results:
            # Extract numbers only
            text = ''.join(results)
            numbers = self.number_pattern.findall(text)
            return numbers[0] if numbers else "0"
        
        return "0"
    
    def read_ammo(self, screenshot) -> int:
        """Read ammo count."""
        text = self.read_region(screenshot, (1700, 900, 1850, 930))
        return int(text.replace(',', '').replace('.', ''))
    
    def read_all(self, screenshot) -> dict:
        """Read all number displays."""
        return {
            'ammo': self.read_ammo(screenshot),
            'rockets': self.read_rockets(screenshot),
            'credits': self.read_credits(screenshot),
            'coords': self.read_coords(screenshot),
        }
```

**Output:**
```python
{
    "ammo": 12500,
    "rockets": 150,
    "credits": 1234567,
    "coords": {"x": 24, "y": 18}
}
```

**When to Use OCR vs Color Scan:**
| Data | Use OCR? | Use Color Scan? | Why |
|------|----------|-----------------|-----|
| HP % | ❌ | ✅ | Bar is visual, not number |
| Ammo count | ✅ | ❌ | Exact number needed |
| Coordinates | ✅ | ❌ | Text on screen |
| Shield % | ❌ | ✅ | Bar is visual |
| Credits | ✅ | ❌ | Exact number |

**Performance Note:**
> OCR is slow (~30-50ms). Run it at **1-2 Hz** (every 0.5-1s), not every frame.
> Ammo/credits don't change every frame, so this is fine.



### Task 4: Cooldown Tracking (Timer-Based)

**Purpose:** Track ability cooldowns without vision

**Method:** Track keypresses + countdown timers

**Our Abilities (Known Cooldowns):**
| Key | Ability | Cooldown (s) |
|-----|---------|--------------|
| 1 | Cloak | 60 |
| 2 | EMP | 30 |
| 3 | Insta-Shield | 45 |
| 4 | Smartbomb | 60 |
| 5 | Energy Leech | 30 |
| Space | Jump | 120 |

**Algorithm:**
```python
class CooldownTracker:
    COOLDOWNS = {
        '1': 60,   # Cloak
        '2': 30,   # EMP
        '3': 45,   # Insta-Shield
        '4': 60,   # Smartbomb
        '5': 30,   # Energy Leech
        'space': 120,  # Jump
    }
    
    def __init__(self):
        self.active_cooldowns = {}
    
    def on_keypress(self, key: str):
        """Called when we press an ability key."""
        if key in self.COOLDOWNS:
            self.active_cooldowns[key] = {
                'started': time.time(),
                'duration': self.COOLDOWNS[key]
            }
    
    def get_cooldown_remaining(self, key: str) -> float:
        """Returns seconds remaining, 0 if ready."""
        if key not in self.active_cooldowns:
            return 0.0
        
        cd = self.active_cooldowns[key]
        elapsed = time.time() - cd['started']
        remaining = cd['duration'] - elapsed
        
        return max(0.0, remaining)
    
    def get_all_cooldowns(self) -> dict:
        """Get all cooldown states."""
        return {
            key: self.get_cooldown_remaining(key)
            for key in self.COOLDOWNS
        }
```

**Output:**
```python
{
    "cloak_cd": 0.0,      # Ready
    "emp_cd": 15.3,       # 15s remaining
    "shield_cd": 0.0,     # Ready
    "smartbomb_cd": 42.1, # 42s remaining
}
```

---

### Task 5: Enemy Ability Detection (Optional, Advanced)

**Purpose:** Detect when enemies use abilities

**Method:** YOLO visual effect detection OR animation frames

**VFX Classes (if implemented):**
| Class | Description | Known Cooldown |
|-------|-------------|----------------|
| `emp_effect` | Blue pulse wave | 30s |
| `cloak_shimmer` | Ship fading out | 60s |
| `shield_bubble` | Orange/Blue bubble | 45s |

**On Detection:**
```python
def on_enemy_ability_detected(enemy_id: int, ability: str):
    """When we SEE an enemy use an ability."""
    known_cooldowns = {
        'emp_effect': 30,
        'cloak_shimmer': 60,
        'shield_bubble': 45,
    }
    
    enemy_cooldowns[enemy_id][ability] = {
        'started': time.time(),
        'duration': known_cooldowns[ability]
    }
```

> ⚠️ **This is OPTIONAL for later.** Core system works without it.

---

## Unified State Builder

Combines all detection sources into a single state:

```python
class UnifiedStateBuilder:
    """Combines YOLO + Color Scan + Timers into unified state."""
    
    def __init__(self):
        self.yolo_detector = YOLODetector("best.pt")
        self.hp_reader = HealthBarReader()
        self.cooldown_tracker = CooldownTracker()
        self.object_tracker = ByteTracker()
    
    def build_state(self, screenshot) -> dict:
        """Build complete game state from all sources."""
        
        # 1. YOLO: Detect objects
        detections = self.yolo_detector.detect(screenshot)
        tracked_objects = self.object_tracker.update(detections)
        
        # 2. Color Scan: Read health bars
        health = self.hp_reader.read_all(screenshot)
        
        # 3. Timers: Get cooldown states
        cooldowns = self.cooldown_tracker.get_all_cooldowns()
        
        return {
            'objects': tracked_objects,
            'player_hp': health['player_hp'],
            'player_shield': health['player_shield'],
            'target_hp': health.get('target_hp'),
            'cooldowns': cooldowns,
            'timestamp': time.time(),
        }
```

---

## Implementation Checklist

### Phase 1: Simplify YOLO (Immediate)
- [ ] Merge NPC classes into single `enemy_npc`
- [ ] Retrain with 6 classes only
- [ ] Test detection accuracy improvement

### Phase 2: Add Color Scan (Quick Win)
- [ ] Create `HealthBarReader` class
- [ ] Define fixed UI regions for 1920x1080
- [ ] Add resolution scaling support
- [ ] Test HP reading accuracy

### Phase 3: Add Cooldown Tracking (Easy)
- [ ] Create `CooldownTracker` class
- [ ] Hook into keypress events in `FilteredRecorder`
- [ ] Add cooldown state to recordings

### Phase 4: Unify Pipeline
- [ ] Create `UnifiedStateBuilder` class
- [ ] Update `StateEncoderV2` to use unified state
- [ ] Test full pipeline end-to-end

### Phase 5: Enemy Abilities (Optional, Later)
- [ ] Train YOLO on VFX classes
- [ ] Add enemy cooldown estimation
- [ ] Integrate with win probability calculation

---

## File Structure

```
darkorbit_bot/
├── vision/                      # NEW: Vision pipeline
│   ├── __init__.py
│   ├── yolo_detector.py         # Simplified YOLO (6 classes)
│   ├── health_reader.py         # Color-based HP reading
│   ├── number_reader.py         # OCR for ammo/coords/credits
│   ├── cooldown_tracker.py      # Timer-based cooldowns
│   ├── unified_state.py         # Combines all sources
│   └── ui_regions.py            # Fixed UI region coordinates
│
└── v2/
    ├── perception/
    │   ├── tracker.py           # ByteTrack (unchanged)
    │   └── state_encoder.py     # Uses unified state
```

---

## Performance Comparison

| Approach | Detection Sources | Accuracy | Speed |
|----------|-------------------|----------|-------|
| **Old: YOLO-only** | 1 (50+ classes) | ~70% | ~15ms |
| **New: Hybrid** | 3 specialized | ~95% | ~12ms |

### Speed Breakdown (Hybrid)

| Component | Time | Runs |
|-----------|------|------|
| YOLO (6 classes) | ~8ms | Every frame |
| Color Scan | ~1ms | Every frame |
| Timer Check | ~0.01ms | Every frame |
| ByteTrack | ~2ms | Every frame |
| **Total** | **~11ms** | |

---

## Summary

| Old Approach | New Approach |
|--------------|--------------|
| YOLO does everything | Specialized tools |
| 50+ classes | 6 YOLO classes |
| No HP reading | Color scan HP |
| No cooldowns | Timer-based CDs |
| Confused/inaccurate | Focused/precise |

---

## Async Architecture (Future: When OCR Added)

### Current: Sync Pipeline (No OCR)

```
Screenshot → YOLO → ByteTrack → ColorScan → Timers → StateEncoder
           10ms     2ms         1ms         0ms      → Total: ~13ms ✅
```

**No async needed.** All components are fast enough for 60 fps.

### Future: Async Pipeline (With OCR)

When OCR is added, it will be too slow to run every frame:

| Component | Time | If Run Every Frame |
|-----------|------|-------------------|
| OCR | ~50ms | Would drop to 14 fps ❌ |

**Solution: Run OCR in background thread with caching.**

```
Main Thread (60 fps)           Background Thread (2 Hz)
────────────────────           ─────────────────────────
Screenshot                     
   ↓                           
YOLO → ByteTrack              OCR (reads screenshot)
   ↓                              ↓
ColorScan                     Updates cached_ocr values
   ↓                              ↓
StateEncoder ← reads cached_ocr ──┘
```

### Async Implementation (When Needed)

```python
class AsyncVisionPipeline:
    def __init__(self):
        self.cached_ocr = {'ammo': 0, 'credits': 0}
        self._ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
        self._ocr_thread.start()
    
    def _ocr_loop(self):
        """Background OCR at 2 Hz."""
        while True:
            if self._screenshot is not None:
                self.cached_ocr = self.ocr.read_all(self._screenshot)
            time.sleep(0.5)
    
    def process(self, screenshot):
        self._screenshot = screenshot  # For OCR thread
        
        # Fast components (blocking, ~13ms total)
        detections = self.yolo.detect(screenshot)
        objects = self.tracker.update(detections)
        health = self.hp_reader.read(screenshot)
        
        # Use cached OCR (non-blocking)
        return {
            'objects': objects,
            'health': health,
            'ammo': self.cached_ocr['ammo'],  # From cache
        }
```

---

## Implementation Phases

| Phase | Components | Async? | Status |
|-------|------------|--------|--------|
| 1 | YOLO + ByteTrack + ColorScan + Timers | No | To implement |
| 2 | Add OCR (ammo, coords) | Yes | Future |
| 3 | Add VFX detection (enemy abilities) | No | Optional |

> **Note:** Start with Phase 1 (sync). Add async only when OCR is needed.

---

## Threading Concerns (For Future Reference)

| Concern | Solution |
|---------|----------|
| Race conditions | Use `threading.Lock()` for shared data |
| Screenshot freshness | OCR reads cached screenshot, not live |
| Thread cleanup | Use `daemon=True` for auto-cleanup |
| Error handling | Wrap OCR in try/except, don't crash main |

---

## Files to Create

### Phase 1 (Sync - Current Priority)
```
vision/
├── __init__.py
├── yolo_detector.py
├── health_reader.py
├── cooldown_tracker.py
├── unified_state.py
└── ui_regions.py
```

### Phase 2 (Async - When OCR Added)
```
vision/
├── number_reader.py      # OCR
├── async_pipeline.py     # Threading wrapper
└── (update unified_state.py to use caching)
```

