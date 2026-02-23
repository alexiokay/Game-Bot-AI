# Codebase Organization Summary

This document describes the cleaned-up project structure.

## What Changed

### Moved to `yolo/autolabeling/`
- `autodistill_autolabel.py` - Grounding DINO auto-labeling
- `autodistill_sam_autolabel.py` - Grounded SAM (DINO + SAM)
- `autolabel_sam_first.py` - SAM-first approach
- `sam_clip_autolabel.py` - SAM + CLIP zero-shot
- `autolabel_with_sam.py` - Basic SAM labeling
- `install_autodistill.bat` - Auto-labeling dependencies

### Moved to `yolo/web_ui/`
- `autolabel_server.py` - Flask backend for web UI
- `interactive_autolabel_web.html` - Fast JavaScript web UI
- `install_flask.bat` - Web UI dependencies
- `WEB_UI_GUIDE.md` - Web UI documentation

### Moved to `yolo/`
- `train_detect.py` - Main training script
- `test_model.py` - Model testing
- `test_all_models.py` - Compare multiple models
- `yolo_finetuning.py` - Fine-tuning script
- `analyze_dataset.py` - Dataset analysis
- `inspect_yolo_model.py` - Model inspection

### Removed (Outdated)
- `test_overfitting.py` - Redundant testing script
- `train_like_working.py` - Old training approach
- `train_seg.py` - Segmentation training (not used)

### Kept in Main Directory
- `interactive_autolabel_gui.py` - Python Tkinter GUI (backup, slower than web UI)
- `auto_screenshot.py` - Screenshot automation
- `fix_labels.py` - Label correction utility
- `test_gui_detector.py` - GUI testing
- `validate_recording.py` - Recording validation
- `semantic_image_search.py` - Image search tool

## New Directory Structure

```
bot/
├── yolo/                          # All YOLO-related work
│   ├── autolabeling/              # Auto-labeling scripts
│   │   ├── autodistill_sam_autolabel.py  (BEST)
│   │   ├── autodistill_autolabel.py
│   │   ├── sam_clip_autolabel.py
│   │   ├── autolabel_sam_first.py
│   │   ├── autolabel_with_sam.py
│   │   ├── install_autodistill.bat
│   │   └── README.md
│   │
│   ├── web_ui/                    # Interactive label review
│   │   ├── autolabel_server.py
│   │   ├── interactive_autolabel_web.html
│   │   ├── install_flask.bat
│   │   └── WEB_UI_GUIDE.md
│   │
│   ├── datasets/                  # YOLO datasets
│   ├── runs/                      # Training runs & models
│   ├── training_screenshots/      # Raw images
│   ├── reviewed_labels/           # Approved labels
│   │
│   ├── train_detect.py            # Main training
│   ├── test_model.py
│   ├── test_all_models.py
│   ├── yolo_finetuning.py
│   ├── analyze_dataset.py
│   ├── inspect_yolo_model.py
│   ├── autolabel_config.json      (auto-generated)
│   └── README.md
│
├── data/                          # Game data
├── models/                        # Other models
├── checkpoints/                   # Model checkpoints
├── runs/                          # Other runs
├── v2/                            # Version 2 code
│
└── (Main directory - utilities)
    ├── interactive_autolabel_gui.py  # Tkinter GUI (fallback)
    ├── auto_screenshot.py
    ├── fix_labels.py
    ├── test_gui_detector.py
    ├── validate_recording.py
    └── semantic_image_search.py
```

## Key Improvements

### 1. Clear Separation of Concerns
- Auto-labeling tools in one place
- Web UI separate from scripts
- Training/testing scripts grouped together

### 2. Model Selection Feature
- Web UI now discovers all trained models automatically
- Switch between models via dropdown
- Selected model saved as default

### 3. Better Documentation
- README in each major directory
- Clear usage instructions
- Workflow guides

### 4. Removed Redundancy
- Deleted outdated/duplicate scripts
- Kept only working versions
- Clear which script to use when

## Recommended Workflow

### For Labeling:
1. Take screenshots → `yolo/training_screenshots/`
2. Auto-label → `cd yolo/autolabeling && python autodistill_sam_autolabel.py`
3. Review → `cd yolo/web_ui && python autolabel_server.py`
4. Export from web UI

### For Training:
1. Prepare dataset → `yolo/datasets/your_dataset/`
2. Train → `cd yolo && python train_detect.py`
3. Models saved to → `yolo/runs/your_run_name/weights/`

### For Testing:
1. Use web UI model selector to test different models
2. Or run `cd yolo && python test_all_models.py`

## Quick Access

- **Web UI**: `cd yolo/web_ui && python autolabel_server.py`
- **Training**: `cd yolo && python train_detect.py`
- **Auto-label**: `cd yolo/autolabeling && python autodistill_sam_autolabel.py`

## Configuration Files

### Auto-generated:
- `yolo/autolabel_config.json` - Stores selected model for web UI

### Manual:
- `yolo/datasets/*/data.yaml` - Dataset configuration
- Training parameters in `train_detect.py`
- Class prompts in auto-labeling scripts

## Notes

- Python Tkinter GUI (`interactive_autolabel_gui.py`) still available but web UI is much faster
- All auto-labeling scripts output YOLO format
- Web UI works with any model in `yolo/runs/`
- Flask installed via `yolo/web_ui/install_flask.bat`
