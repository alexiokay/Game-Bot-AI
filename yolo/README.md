# YOLO Training & Auto-Labeling

This directory contains all YOLO-related scripts and tools for the DarkOrbit bot.

## Directory Structure

```
yolo/
├── autolabeling/          # Auto-labeling scripts (SAM, Grounding DINO, etc.)
├── web_ui/                # Interactive web UI for label review
├── datasets/              # YOLO datasets (images + labels)
├── runs/                  # Training runs and model weights
├── training_screenshots/  # Raw screenshots to label
└── reviewed_labels/       # Manually reviewed and approved labels
```

## Quick Start

### 1. Train a Model

```bash
cd F:/dev/bot/yolo
python train_detect.py
```

This trains YOLO26m on your dataset with optimized settings:
- 1280px resolution (best for small objects)
- batch=4 (prevents GPU overload)
- Early stopping after 20 epochs without improvement

### 2. Review Labels with Web UI

```bash
cd F:/dev/bot/yolo/web_ui
python autolabel_server.py
```

Then open http://localhost:5000 in your browser to:
- Select which trained model to use
- Adjust confidence thresholds per class
- Accept/reject labels interactively
- Export approved labels

### 3. Auto-Label New Screenshots

```bash
cd F:/dev/bot/yolo/autolabeling
python autodistill_sam_autolabel.py
```

Uses Grounding DINO + SAM (Roboflow's method) to auto-generate labels.

## Main Scripts

### Training
- **train_detect.py** - Main training script (YOLO26m, 1280px, optimized)
- **yolo_finetuning.py** - Fine-tuning script for existing models
- **test_model.py** - Test model performance on validation set
- **test_all_models.py** - Compare multiple models

### Analysis
- **analyze_dataset.py** - Analyze dataset statistics and class distribution
- **inspect_yolo_model.py** - Inspect model architecture and classes

## Configuration

### Dataset Format

YOLO format:
```
datasets/your_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### Training Settings

Current optimal settings for DarkOrbit (small objects):
- **Model**: YOLO26m (medium variant)
- **Resolution**: 1280px (4x more pixels than 640px)
- **Batch size**: 4 (fits in GPU memory)
- **Epochs**: 100 (with early stopping)
- **Workers**: 0 (Windows fix)

## Workflows

### Creating New Labels

1. Take screenshots → `training_screenshots/`
2. Auto-label → `cd autolabeling && python autodistill_sam_autolabel.py`
3. Review in web UI → `cd web_ui && python autolabel_server.py`
4. Export approved labels → Click "Export Labels" in web UI
5. Merge with dataset → Copy from `reviewed_labels/` to `datasets/`
6. Train model → `python train_detect.py`

### Testing Different Models

1. Train multiple models with different settings
2. Run `python test_all_models.py` to compare performance
3. Select best model in web UI for label review

## Model Selection

The web UI automatically discovers all trained models in `runs/`:
- Looks for `best.pt` (or `last.pt` if best doesn't exist)
- Saves your selected model as default
- Switch models anytime via dropdown

## Tips

- **Start with auto-labeling**: Use SAM + Grounding DINO for initial labels
- **Review everything**: Always review auto-labels before training
- **Small dataset?**: Use transfer learning (COCO-pretrained models)
- **GPU overload?**: Reduce batch size or resolution
- **Poor accuracy?**: Collect more diverse training data

## Troubleshooting

### "No models found"
- Train a model first: `python train_detect.py`
- Check `runs/` directory has folders with `weights/best.pt`

### GPU memory errors
- Reduce batch size in train_detect.py
- Lower resolution (1280 → 960 → 640)

### Flask not found
```bash
cd web_ui
uv pip install flask flask-cors pillow
```

## Next Steps

1. Train your first model if you haven't already
2. Try the web UI to review labels
3. Experiment with auto-labeling different class prompts
4. Iterate: train → test → improve data → train again
