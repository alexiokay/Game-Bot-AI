# Auto-Labeling Tools

This directory contains all auto-labeling scripts for generating YOLO labels automatically.

## Scripts

### Grounding DINO (Roboflow Method)
- **autodistill_autolabel.py** - Uses Grounding DINO for text-based zero-shot detection
- **autodistill_sam_autolabel.py** - Grounding DINO + SAM for instance segmentation (best quality)

### SAM-Based Methods
- **autolabel_sam_first.py** - SAM-first approach: segment everything, then classify
- **sam_clip_autolabel.py** - SAM + CLIP for zero-shot segmentation and classification
- **autolabel_with_sam.py** - Basic SAM auto-labeling

## Installation

Run `install_autodistill.bat` to install dependencies:
```bash
uv pip install autodistill autodistill-grounding-dino autodistill-grounded-sam
```

## Recommended Approach

For best results, use **autodistill_sam_autolabel.py** (Grounded SAM):
- Combines Grounding DINO's text-based detection with SAM's precise segmentation
- This is what Roboflow uses internally
- Outputs polygon format (more accurate than bounding boxes)

## Usage

All scripts follow the same pattern:

```python
# 1. Define class prompts
class_prompts = {
    "enemy-ship": "red enemy spaceship",
    "player-ship": "blue player spaceship",
    # ...
}

# 2. Run auto-labeling
autolabel_with_grounded_sam(
    input_dir='F:/dev/bot/yolo/training_screenshots',
    output_dir='F:/dev/bot/yolo/autolabeled_grounded_sam',
    class_prompts=class_prompts,
    box_threshold=0.30,
    text_threshold=0.25
)
```

## Output

All scripts output YOLO format:
```
output_dir/
├── images/
│   └── *.jpg
└── labels/
    └── *.txt
```

## Next Steps

After auto-labeling:
1. Review labels using the interactive web UI (see parent directory)
2. Manually correct any errors in Roboflow
3. Merge with existing dataset
4. Retrain YOLO model
