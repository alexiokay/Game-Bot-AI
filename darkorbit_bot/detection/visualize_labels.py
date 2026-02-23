import cv2
import os
import argparse
from pathlib import Path
import yaml
import json
import random

def load_class_names(data_dir: Path):
    """Try to load class names from dataset.yaml or ui_vocabulary.json"""
    # Try dataset.yaml first
    yaml_path = data_dir / "yolo_dataset" / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                return data['names']
    
    # Fallback to vocabulary
    vocab_path = data_dir / "ui_vocabulary.json"
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            names = {}
            idx = 0
            for obj in vocab.get('classes', {}).get('game_objects', []):
                names[idx] = obj['name']
                idx += 1
            for elem in vocab.get('classes', {}).get('ui_elements', []):
                names[idx] = elem['name']
                idx += 1
            return names
            
    return {}

def get_color(class_id):
    random.seed(class_id)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def visualize(image_dir: Path, label_dir: Path, output_dir: Path, class_names: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    print(f"Processing {len(images)} images from {image_dir}...")
    
    count = 0
    for img_path in images:
        # distinct labels dir provided?
        if label_dir:
            txt_path = label_dir / f"{img_path.stem}.txt"
        else:
            # Assume labels are in sibling folder or inferred
            # Try 1: ../labels/same_subfolder/name.txt
             if "images" in str(img_path.parent):
                 # e.g. .../images/train/foo.png -> .../labels/train/foo.txt
                 txt_path = Path(str(img_path.parent).replace("images", "labels")) / f"{img_path.stem}.txt"
             else:
                 # sibling
                 txt_path = img_path.with_suffix(".txt")
                 
        if not txt_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        found_box = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                found_box = True
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # De-normalize
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                name = class_names.get(cls_id, str(cls_id))
                color = get_color(cls_id)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Label background
                label_size, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        if found_box:
            out_path = output_dir / f"vis_{img_path.name}"
            cv2.imwrite(str(out_path), img)
            count += 1
            
    print(f"✅ Saved {count} visualized images to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO Labels')
    parser.add_argument('--dir', type=str, required=True, help='Image directory')
    parser.add_argument('--labels', type=str, help='Label directory (optional, tries to infer)')
    args = parser.parse_args()
    
    img_dir = Path(args.dir)
    if args.labels:
        lbl_dir = Path(args.labels)
    else:
        lbl_dir = None # Will infer
        
    out_dir = img_dir / "preview_boxes"
    
    # Helper to find data dir roughly
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    names = load_class_names(data_dir)
    
    visualize(img_dir, lbl_dir, out_dir, names)

if __name__ == "__main__":
    main()
