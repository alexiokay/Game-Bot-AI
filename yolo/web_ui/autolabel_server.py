"""Flask backend server for interactive auto-labeling web UI.

Serves YOLO detection results to the web frontend.

Usage:
    python autolabel_server.py
    Then open http://localhost:5000 in browser
"""
import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Configuration
RUNS_DIR = Path('F:/dev/bot/yolo/runs')
OUTPUT_DIR = Path('F:/dev/bot/yolo/reviewed_labels')
CONFIG_FILE = Path('F:/dev/bot/yolo/autolabel_config.json')

# Global state
model = None
CLASS_COLORS = {}
current_model_path = None
INPUT_DIR = Path('F:/dev/bot/yolo/training_screenshots')  # Can be changed via API


def discover_models():
    """Find all trained YOLO models in runs directory."""
    models = []
    if not RUNS_DIR.exists():
        return models

    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir():
            weights_dir = run_dir / 'weights'
            if weights_dir.exists():
                best_pt = weights_dir / 'best.pt'
                last_pt = weights_dir / 'last.pt'

                if best_pt.exists():
                    models.append({
                        'name': run_dir.name,
                        'path': str(best_pt),
                        'type': 'best'
                    })
                elif last_pt.exists():
                    models.append({
                        'name': run_dir.name,
                        'path': str(last_pt),
                        'type': 'last'
                    })

    return sorted(models, key=lambda x: x['name'])


def load_config():
    """Load saved configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save configuration."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def load_model(model_path: str):
    """Load YOLO model and initialize class colors."""
    global model, CLASS_COLORS, current_model_path

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    current_model_path = model_path

    # Generate consistent colors for classes
    CLASS_COLORS = {}
    np.random.seed(42)
    for i, class_name in enumerate(model.names.values()):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        CLASS_COLORS[class_name] = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'

    print(f"Model loaded: {len(model.names)} classes")

    # Save as default
    save_config({'default_model': model_path})


# Try to load saved configuration
config = load_config()

# Load saved input directory
saved_input_dir = config.get('input_dir')
if saved_input_dir and Path(saved_input_dir).exists():
    INPUT_DIR = Path(saved_input_dir)
    print(f"Loaded saved input directory: {INPUT_DIR}")

# Load default model
default_model = config.get('default_model')

if default_model and Path(default_model).exists():
    load_model(default_model)
else:
    # Try to find any model
    available_models = discover_models()
    if available_models:
        load_model(available_models[0]['path'])
        print(f"Loaded first available model: {available_models[0]['name']}")
    else:
        print("WARNING: No models found! Please train a model first.")
        print(f"Looking in: {RUNS_DIR}")


@app.route('/')
def index():
    """Serve the HTML frontend."""
    return send_file('interactive_autolabel_web.html')


@app.route('/images', methods=['GET'])
def get_images():
    """Get list of all images in input directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        str(p.relative_to(INPUT_DIR))
        for p in INPUT_DIR.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    return jsonify(sorted(images))


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all class names."""
    if model is None:
        return jsonify({'error': 'No model loaded'}), 500
    return jsonify(list(model.names.values()))


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    models = discover_models()
    return jsonify({
        'models': models,
        'current': current_model_path
    })


@app.route('/models/select', methods=['POST'])
def select_model():
    """Select and load a different model."""
    data = request.json
    model_path = data.get('model_path')

    if not model_path or not Path(model_path).exists():
        return jsonify({'error': 'Model not found'}), 404

    try:
        load_model(model_path)
        return jsonify({
            'success': True,
            'model_path': model_path,
            'classes': list(model.names.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/browse_folders', methods=['POST'])
def browse_folders():
    """Browse folders server-side."""
    data = request.json
    current_path = data.get('path', 'F:/dev/bot/yolo')

    try:
        path = Path(current_path)
        if not path.exists() or not path.is_dir():
            path = Path('F:/dev/bot/yolo')

        # Get subdirectories
        folders = []
        try:
            for item in sorted(path.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    # Count images in folder
                    image_count = len([f for f in item.rglob('*')
                                     if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
                    folders.append({
                        'name': item.name,
                        'path': str(item),
                        'image_count': image_count
                    })
        except PermissionError:
            pass

        # Get parent directory
        parent = str(path.parent) if path.parent != path else None

        return jsonify({
            'current': str(path),
            'parent': parent,
            'folders': folders
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/set_folder', methods=['POST'])
def set_folder():
    """Change the input folder for images."""
    global INPUT_DIR

    data = request.json
    folder_path = data.get('folder_path')

    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400

    folder = Path(folder_path)
    if not folder.exists():
        return jsonify({'error': 'Folder does not exist'}), 404

    if not folder.is_dir():
        return jsonify({'error': 'Path is not a directory'}), 400

    INPUT_DIR = folder

    # Save to config
    config = load_config()
    config['input_dir'] = str(folder)
    save_config(config)

    return jsonify({
        'success': True,
        'folder': str(folder),
        'image_count': len(get_images_list())
    })


@app.route('/image/<int:index>', methods=['GET'])
def get_image(index: int):
    """Get image by index."""
    images = get_images_list()
    if 0 <= index < len(images):
        image_path = INPUT_DIR / images[index]
        return send_file(image_path, mimetype='image/jpeg')
    return jsonify({'error': 'Image not found'}), 404


@app.route('/detect', methods=['POST'])
def detect():
    """Run YOLO detection on image with custom thresholds.

    Request body:
        {
            "image_path": "relative/path/to/image.jpg",
            "thresholds": {"class1": 0.3, "class2": 0.4, ...}
        }

    Returns:
        {
            "image": "base64_encoded_image",
            "detections": [
                {
                    "class": "enemy-ship",
                    "confidence": 0.85,
                    "bbox": [x1, y1, x2, y2],
                    "color": "#ff0000"
                },
                ...
            ],
            "image": {"width": 1920, "height": 1080}
        }
    """
    if model is None:
        return jsonify({'error': 'No model loaded'}), 500

    data = request.json
    image_path = INPUT_DIR / data['image_path']
    thresholds = data.get('thresholds', {})

    # Run YOLO detection with lowest threshold (we'll filter on frontend)
    min_threshold = min(thresholds.values()) if thresholds else 0.01
    results = model.predict(
        source=str(image_path),
        conf=min_threshold,
        verbose=False
    )[0]

    # Load image to get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Convert RGBA to RGB if necessary (for PNG with transparency)
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Encode image as base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Extract detections
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'color': CLASS_COLORS[class_name]
            })

    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'detections': detections,
        'image_size': {'width': img_width, 'height': img_height}
    })


@app.route('/export', methods=['POST'])
def export_labels():
    """Export accepted labels to YOLO format.

    Request body:
        {
            "decisions": {"0": "accept", "5": "reject", ...},
            "thresholds": {"class1": 0.3, "class2": 0.4, ...}
        }

    Returns:
        {
            "count": 42,
            "output_dir": "/path/to/output"
        }
    """
    data = request.json
    decisions = data['decisions']
    thresholds = data['thresholds']

    # Create output directories
    output_images = OUTPUT_DIR / 'images'
    output_labels = OUTPUT_DIR / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = get_images_list()
    accepted_count = 0

    for index_str, decision in decisions.items():
        if decision != 'accept':
            continue

        index = int(index_str)
        if index >= len(images):
            continue

        image_path = INPUT_DIR / images[index]

        # Run detection
        results = model.predict(
            source=str(image_path),
            conf=min(thresholds.values()) if thresholds else 0.01,
            verbose=False
        )[0]

        # Filter by thresholds
        img = Image.open(image_path)
        img_width, img_height = img.size

        filtered_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if confidence >= thresholds.get(class_name, 0.3):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Convert to YOLO format (normalized center coords + width/height)
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    filtered_boxes.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Skip if no detections
        if not filtered_boxes:
            continue

        # Copy image
        output_image_path = output_images / image_path.name
        img.save(output_image_path)

        # Write label file
        output_label_path = output_labels / f"{image_path.stem}.txt"
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(filtered_boxes))

        accepted_count += 1

    return jsonify({
        'count': accepted_count,
        'output_dir': str(OUTPUT_DIR)
    })


def get_images_list() -> List[str]:
    """Helper to get sorted list of images."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        str(p.relative_to(INPUT_DIR))
        for p in INPUT_DIR.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    return sorted(images)


if __name__ == '__main__':
    print("=" * 60)
    print("  INTERACTIVE AUTO-LABEL WEB SERVER")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Runs directory: {RUNS_DIR}")

    if model and current_model_path:
        print(f"Model: {current_model_path}")
        print(f"Classes: {len(model.names)} ({', '.join(list(model.names.values())[:5])}...)")
    else:
        print("WARNING: No model loaded!")

    print(f"\nOpen in browser: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, port=5000, host='0.0.0.0')
