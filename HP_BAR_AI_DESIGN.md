# AI-Learned HP Bar Reading Design

## Overview
Instead of using hardcoded OpenCV color thresholds, we'll leverage V2's existing visual infrastructure to **learn** how to read HP/shield bars from pixels. This aligns with the V2 philosophy of learned behavior over hand-crafted rules.

## Why AI-Learned Approach?

### Problems with Rule-Based (OpenCV):
- **Brittle**: Breaks with different enemy types (bosses have different bars)
- **Unreliable**: Visual effects (explosions, lasers) interfere with color detection
- **Inaccurate**: Lighting, zoom level affect color values
- **Not adaptable**: Can't handle new enemy types without manual tweaking

### Benefits of AI-Learned:
- **Robust**: Learns invariant features that work across different enemies
- **Adaptive**: Can generalize to new enemy types
- **Precise**: Can learn fine-grained HP percentages (not just HIGH/MED/LOW)
- **Integrated**: Uses existing V2 visual infrastructure (no extra code)

## Architecture Options

### **Option 1: HP Regression Head (RECOMMENDED)**
Add a dedicated neural network head that predicts HP% from enemy RoI visual features.

**How it works:**
```
Enemy RoI → VisionBackbone (MobileNetV3) → HP Regression Head → HP% (0.0-1.0)
                                          ↘ Tactician RoI features (existing)
```

**Advantages:**
- Fast inference (1-2ms per enemy, runs in parallel with existing RoI encoding)
- Precise percentage output (0-100%)
- Easy to train with supervised learning
- Minimal integration complexity

**Training:**
- Collect frames with HP bars + manual labels or OCR-based labels
- Train small regression head (~100k params) on top of frozen backbone
- Loss: MSE or Huber loss between predicted and true HP%

**Architecture:**
```python
class HPBarReader(nn.Module):
    def __init__(self):
        self.vision_backbone = VisionBackbone(config)  # Shared with Tactician

        # HP-specific head
        self.hp_head = nn.Sequential(
            nn.Linear(576, 256),  # MobileNetV3 features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),     # Output: HP percentage [0-1]
            nn.Sigmoid()
        )

        # Optional: Multi-task with shield reading
        self.shield_head = nn.Sequential(...)

    def forward(self, roi_features):
        """
        Args:
            roi_features: [B, 576] from VisionBackbone
        Returns:
            hp_percent: [B, 1] HP percentage 0.0-1.0
        """
        return self.hp_head(roi_features)
```

**Integration:**
```python
# In VisionEncoder.encode_rois()
roi_features_backbone = self.backbone.forward_backbone(batch)  # [N, 576]
roi_features_tactician = self.backbone.roi_proj(roi_features_backbone)  # [N, 128]

# NEW: HP bar reading
if self.hp_reader is not None:
    hp_percentages = self.hp_reader(roi_features_backbone)  # [N, 1]
else:
    hp_percentages = None

return {
    'tactician_features': roi_features_tactician,
    'hp_percentages': hp_percentages
}
```

---

### **Option 2: Multi-Task Learning**
Train the existing VisionBackbone with an auxiliary HP prediction task.

**How it works:**
```
Enemy RoI → VisionBackbone → [Tactician Features, HP%, Shield%]
```

**Advantages:**
- HP reading improves Tactician's object representation
- Single forward pass for both tasks
- Visual features become HP-aware

**Disadvantages:**
- Requires retraining entire VisionBackbone
- More complex training pipeline
- Risk of catastrophic forgetting if not done carefully

---

### **Option 3: VLM-Based HP Reading**
Use the existing VLM (Vision-Language Model) to analyze HP bars via prompting.

**How it works:**
```python
prompt = "What percentage of HP does this enemy have? Answer with just a number 0-100."
hp_percent = vlm.query(enemy_roi_image, prompt)
```

**Advantages:**
- Zero training required
- Can handle ANY visual element (HP, shields, buffs, debuffs)
- Natural language interface

**Disadvantages:**
- Extremely slow (100-500ms per VLM call)
- Can't run at 60fps (would need caching/batching)
- Overkill for a simple regression task

---

### **Option 4: Embed HP Features in Visual Encoder**
Modify VisionBackbone to explicitly extract HP bar region features.

**How it works:**
```python
# VisionBackbone learns to attend to HP bar region
roi_features = vision_encoder.encode_roi(enemy_bbox)
# roi_features[0:64] = object appearance
# roi_features[64:96] = HP bar features (learned implicitly)
```

**Advantages:**
- HP information flows naturally into Tactician decisions
- No separate HP prediction step
- Model learns what's relevant

**Disadvantages:**
- Implicit - harder to verify HP reading accuracy
- Requires labeled training data with HP values
- Less interpretable

---

## Recommended Approach: **Option 1 (HP Regression Head)**

**Why?**
- Fast (1-2ms per enemy)
- Precise (0-100% HP)
- Easy to train and validate
- Minimal changes to existing codebase
- Can be added incrementally

**Implementation Plan:**

### Step 1: Data Collection
Create a tool to collect HP bar training data:

```python
# tools/collect_hp_labels.py
# - Capture enemy RoIs
# - Display frame with HP bar
# - User inputs HP% (or auto-extract via OCR if visible)
# - Save: {roi_image, enemy_class, hp_percent, shield_percent}
```

**Target dataset size:** 5,000-10,000 labeled RoIs across different enemy types

### Step 2: Train HP Regression Head

```python
# training/train_hp_reader.py
# 1. Load VisionBackbone (frozen)
# 2. Train HP head with MSE loss
# 3. Validate on held-out enemies
# 4. Export hp_reader.pt
```

**Training loss:**
```python
# Huber loss (robust to outliers)
loss = F.huber_loss(predicted_hp, target_hp)

# Or weighted MSE (prioritize low HP accuracy)
weights = 1.0 + 2.0 * (1.0 - target_hp)  # Low HP = higher weight
loss = (weights * (predicted_hp - target_hp)**2).mean()
```

### Step 3: Integrate into VisionEncoder

Modify `vision_encoder.py`:

```python
class VisionEncoder:
    def __init__(self, config):
        # Existing code...

        # Load HP reader if available
        self.hp_reader = None
        hp_reader_path = Path(__file__).parent / 'hp_reader.pt'
        if hp_reader_path.exists():
            self.hp_reader = HPBarReader().to(self.device)
            self.hp_reader.load_state_dict(torch.load(hp_reader_path))
            self.hp_reader.eval()
            logger.info("HP bar reader loaded")

    @torch.no_grad()
    def encode_rois(self, frame, bboxes):
        # ... existing RoI encoding ...

        # Extract HP percentages
        hp_percentages = np.zeros(self.config.max_rois, dtype=np.float32)

        if self.hp_reader is not None and len(roi_tensors) > 0:
            batch = torch.stack(roi_tensors).to(self.device, dtype=self.dtype)
            backbone_features = self.backbone.forward_backbone(batch)
            hp_preds = self.hp_reader(backbone_features)
            hp_percentages[:len(hp_preds)] = hp_preds.cpu().float().numpy().flatten()

        return {
            'roi_features': features,
            'hp_percentages': hp_percentages
        }
```

### Step 4: Expose HP Info to Tactician

Modify `state_encoder.py` to include HP in object features:

```python
# In StateEncoderV2.encode()
if vision_results and 'hp_percentages' in vision_results:
    hp_values = vision_results['hp_percentages']

    # Add HP as object feature [dimension 20 → 21]
    for i, obj in enumerate(tracked_objects):
        if i < len(hp_values):
            obj_features[i, 20] = hp_values[i]  # Enemy HP%
```

### Step 5: Use HP for Target Prioritization

The Tactician can now use enemy HP for smarter targeting:
- Finish low-HP enemies first (prevent healing/escape)
- Avoid high-HP bosses when low on ammo
- Focus damaged enemies in group fights

**No code changes needed** - Tactician learns priorities from demonstrations!

---

## Performance Analysis

### Inference Speed (per frame with 10 enemies):

**Option 1 (HP Regression Head):**
```
VisionBackbone forward: 8ms (shared with Tactician RoI encoding)
HP head forward (10 enemies): 1-2ms
Total overhead: ~1-2ms (HP head only, backbone already runs)
```

**Option 2 (Multi-Task):**
```
Same as Option 1 (0ms overhead, already computing roi features)
```

**Option 3 (VLM):**
```
VLM forward per enemy: 100-500ms
Not feasible for real-time
```

### Accuracy Expectations:

**With 10k training samples:**
- HP% MAE (Mean Absolute Error): ±3-5%
- Low HP (<20%) MAE: ±2-3% (with weighted loss)
- Cross-enemy generalization: 85-90% (depends on visual variety)

**With 50k training samples:**
- HP% MAE: ±2-3%
- Low HP MAE: ±1-2%
- Cross-enemy generalization: 95%+

---

## Training Data Collection Strategy

### Manual Labeling (Fast Start):
```python
# tools/hp_labeler.py
# 1. Run bot in recording mode
# 2. Every 1 second, save enemy RoIs
# 3. Display RoI + HP bar region
# 4. User clicks on HP bar → auto-calculate HP% from click position
# 5. Save labeled data

# Expected speed: 500-1000 labels/hour
# Target: 5000 labels = 5-10 hours of labeling
```

### Semi-Supervised (Scale Up):
```python
# 1. Train initial model on 5k manual labels
# 2. Run bot with HP reader
# 3. Auto-label easy cases (high confidence predictions)
# 4. Human verify borderline cases
# 5. Retrain with expanded dataset

# Can scale to 50k+ labels with minimal effort
```

### Synthetic Data Augmentation:
```python
# Generate synthetic HP bars:
# - Different bar lengths
# - Different colors (green→yellow→red gradient)
# - Add noise, blur, lighting variations
# - Overlay on enemy RoIs

# Can generate unlimited training data
```

---

## Alternative: Hybrid Approach

**Fast Prototype:** Use lightweight color detection for now, collect data while bot runs

```python
class HybridHPReader:
    def __init__(self):
        self.color_detector = FastColorBasedHP()  # Fallback
        self.ai_reader = HPBarReader() if exists else None
        self.training_buffer = []  # Collect data for future training

    def read_hp(self, roi):
        # Use color detector
        hp_color = self.color_detector.estimate_hp(roi)

        # Save for training dataset
        if random.random() < 0.01:  # 1% sampling
            self.training_buffer.append({
                'roi': roi,
                'hp_estimate': hp_color  # Noisy label
            })

        # Use AI if available (more accurate)
        if self.ai_reader:
            hp_ai = self.ai_reader(roi)
            return hp_ai
        else:
            return hp_color
```

---

## Implementation Timeline

### Phase 1: Data Collection Tool (1-2 hours)
- [hp_labeler.py](f:\dev\bot\tools\hp_labeler.py) - GUI for labeling HP bars
- Save format: `{roi: np.array, hp: float, shield: float, enemy_class: str}`

### Phase 2: HP Reader Training (2-3 hours)
- [train_hp_reader.py](f:\dev\bot\training\train_hp_reader.py) - Training script
- [hp_reader.py](f:\dev\bot\darkorbit_bot\v2\models\hp_reader.py) - Model definition
- Collect 5k labels → Train → Validate

### Phase 3: Integration (1 hour)
- Modify [vision_encoder.py](f:\dev\bot\darkorbit_bot\v2\perception\vision_encoder.py)
- Add HP output to `encode_rois()`
- Update object features in [state_encoder.py](f:\dev\bot\darkorbit_bot\v2\perception\state_encoder.py)

### Phase 4: Testing (1 hour)
- Run bot with HP reader
- Verify HP% accuracy
- Check inference speed

**Total: 5-7 hours**

---

## Next Steps

1. **Confirm approach** - Do you want to proceed with Option 1 (HP Regression Head)?
2. **Create data collection tool** - GUI for labeling enemy HP bars
3. **Collect initial dataset** - 5k labeled RoIs
4. **Train HP reader model**
5. **Integrate and test**

Would you like me to start implementing the data collection tool?
