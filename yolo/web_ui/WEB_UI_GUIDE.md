# Interactive Auto-Label Web UI

Fast JavaScript-based UI for reviewing and adjusting YOLO auto-labels in real-time.

## Why JavaScript UI?

The web UI is **much faster** than the Python Tkinter version because:
- **Non-blocking updates**: JavaScript runs in event loop, slider updates don't freeze UI
- **Instant rendering**: Canvas updates happen immediately without waiting for Python
- **Smooth interactions**: No debouncing needed, handles 60+ updates/second easily
- **Modern browser optimizations**: Hardware-accelerated canvas rendering

## Installation

1. Navigate to web_ui directory:
```bash
cd F:/dev/bot/yolo/web_ui
```

2. Install Flask dependencies:
```bash
install_flask.bat
```

Or manually:
```bash
uv pip install flask flask-cors pillow
```

## Usage

1. **Start the server:**
```bash
cd F:/dev/bot/yolo/web_ui
python autolabel_server.py
```

2. **Open in browser:**
```
http://localhost:5000
```

3. **Select Model:**
   - Use the dropdown at the top to select which trained YOLO model to use
   - The server will automatically discover all models in `F:/dev/bot/yolo/runs/`
   - Your selection is saved as the default for next time

4. **Review labels:**
   - Adjust global confidence slider (updates ALL classes instantly)
   - Or adjust individual class sliders for fine control
   - Use arrow keys (← →) or buttons to navigate
   - Press 'A' to accept, 'R' to reject
   - Detections update in real-time as you adjust sliders

5. **Export accepted labels:**
   - Click "Export Labels" button
   - Accepted labels saved to `F:/dev/bot/yolo/reviewed_labels/`

## Features

### Per-Class Confidence Thresholds
- Each class has its own confidence slider (0.00 - 1.00)
- Adjust thresholds to filter out false positives
- Changes apply instantly to the image preview

### Global Slider
- Adjust all class thresholds at once
- Perfect for quick filtering
- Updates all 50+ classes instantly (JavaScript speed!)

### Keyboard Shortcuts
- `←` Previous image
- `→` Next image
- `A` Accept current labels
- `R` Reject current labels

### Statistics
- Real-time tracking of accepted/rejected/pending images
- Progress indicator showing current image number

### Auto-scaling
- Image automatically fits window size
- Responsive layout adapts to browser window
- Canvas updates on window resize

## Configuration

Edit [autolabel_server.py](autolabel_server.py:11-13) to change paths:

```python
INPUT_DIR = Path('F:/dev/bot/yolo/training_screenshots')
MODEL_PATH = 'F:/dev/bot/yolo/runs/darkorbit_v6_yolo26_detect/weights/best.pt'
OUTPUT_DIR = Path('F:/dev/bot/yolo/reviewed_labels')
```

## Output Format

Exported labels are in YOLO format:
```
F:/dev/bot/yolo/reviewed_labels/
├── images/
│   ├── screenshot_001.jpg
│   ├── screenshot_002.jpg
│   └── ...
└── labels/
    ├── screenshot_001.txt
    ├── screenshot_002.txt
    └── ...
```

Each label file contains:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized (0.0 - 1.0).

## Performance Comparison

### Python Tkinter UI
- Global slider: 500-1000ms update time (with debouncing)
- Must debounce to prevent freezing
- Single-threaded blocking operations
- Max ~10 updates/second

### JavaScript Web UI
- Global slider: <16ms update time (instant)
- No debouncing needed
- Event-driven non-blocking
- Handles 60+ updates/second smoothly

## Troubleshooting

### Port 5000 already in use
Change port in [autolabel_server.py](autolabel_server.py:227):
```python
app.run(debug=True, port=5001, host='0.0.0.0')
```

### Model not found
Make sure you've trained a model first:
```bash
python train_detect.py
```

### Images not loading
Check `INPUT_DIR` path is correct and contains images.

### CORS errors
Flask-CORS is installed and configured, but if you still have issues, check browser console for details.
