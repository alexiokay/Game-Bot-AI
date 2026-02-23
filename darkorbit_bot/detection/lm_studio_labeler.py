
import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI
import threading
import concurrent.futures

# Configuration for LM Studio
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"  # Not needed usually

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r') as f:
        return json.load(f)

def build_prompt_from_vocab(vocab):
    # Simplify prompt for local models (they get confused by long definitions)
    prompt = "Detect objects in this game screenshot. Provide bounding boxes [x_min, y_min, x_max, y_max] (0-100 scale).\n"
    prompt += "Look for these classes:\n"
    
    # Just list names, no descriptions (saves context, reduces confusion)
    classes = []
    for obj in vocab.get('classes', {}).get('game_objects', []):
        classes.append(obj['name'])
    for elem in vocab.get('classes', {}).get('ui_elements', []):
        classes.append(elem['name'])
        
    prompt += ", ".join(classes[:50]) + "...\n" # Limit to first 50 common ones to avoid overload?
    # Actually, pass all but just names
    prompt += ", ".join(classes)
    
    prompt += """
\nReturn a JSON array of detected objects. Only list objects that are VISIBLE.
Example: [{"class": "ship", "box": [50, 50, 60, 60]}]
"""
    return prompt

def build_class_map(vocab):
    class_map = {}
    idx = 0
    for obj in vocab.get('classes', {}).get('game_objects', []):
        class_map[obj['name'].lower()] = idx
        idx += 1
    for elem in vocab.get('classes', {}).get('ui_elements', []):
        class_map[elem['name'].lower()] = idx
        idx += 1
    return class_map

def label_with_lm_studio(client, image_path, prompt):
    base64_image = encode_image(image_path)
    
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=4000 # Increased for verbose models
        )
        
        content = response.choices[0].message.content
        
        # Robust JSON extraction
        import re
        import json
        
        # Try to find valid JSON array
        # First, strip markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
            
        # Try fixing truncated JSON (common with local models)
        # If it ends with , or }, try adding ]
        content = content.strip()
        if content.endswith(","):
            content = content[:-1]
        if not content.endswith("]"):
            content += "]"
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
            
        # Fallback: Extract objects with regex one by one
        # {"class": "...", "box": [...]}
        objects = []
        # Regex to find JSON objects
        pattern = r'\{\s*"class"\s*:\s*"([^"]+)"\s*,\s*"box"\s*:\s*\[([\d\s,]+)\]\s*\}'
        matches = re.finditer(pattern, content)
        for m in matches:
            cls = m.group(1)
            box_str = m.group(2)
            box = [float(x) for x in box_str.split(',') if x.strip()]
            if len(box) == 4:
                objects.append({"class": cls, "box": box})
        
        if objects:
            return objects
            
        # Raise primitive error if nothing worked
        raise ValueError("Could not parse JSON")
        
    except Exception as e:
        print(f"\n❌ Error on {Path(image_path).name}: {e}")
        if 'content' in locals() and len(content) > 0:
             print(f"   Raw response start: {content[:100]}...")
             print(f"   Raw response end: ...{content[-100:]}")
        return []

def normalize_box(box, width, height):
    """Normalize box to 0-1 range based on detected scale"""
    if not box or len(box) != 4:
        return [0,0,0,0]
        
    x_min, y_min, x_max, y_max = box
    max_val = max(x_min, y_min, x_max, y_max)
    
    # Heuristic detection
    if max_val <= 1.0: 
        # Already 0-1
        scale_x, scale_y = 1.0, 1.0
    elif max_val <= 100.0:
        # 0-100 scale
        scale_x, scale_y = 100.0, 100.0
    elif max_val <= 1000.0:
        # 0-1000 scale
        scale_x, scale_y = 1000.0, 1000.0
    else:
        # Likely pixels
        scale_x, scale_y = width, height
        
    nx1 = x_min / scale_x
    ny1 = y_min / scale_y
    nx2 = x_max / scale_x
    ny2 = y_max / scale_y
    
    # Clamp
    nx1 = max(0.0, min(1.0, nx1))
    ny1 = max(0.0, min(1.0, ny1))
    nx2 = max(0.0, min(1.0, nx2))
    ny2 = max(0.0, min(1.0, ny2))
    
    return [nx1, ny1, nx2, ny2]

def convert_to_yolo(detections, class_map, width, height):
    lines = []
    for det in detections:
        class_name = det.get('class', '').lower()
        if class_name not in class_map:
            continue
            
        class_id = class_map[class_name]
        raw_box = det.get('box', [0,0,0,0])
        
        # Normalize
        x1, y1, x2, y2 = normalize_box(raw_box, width, height)
        
        # To YOLO Center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        if w > 0 and h > 0:
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
    return lines

def preview_single_image(client, image_path, prompt, class_map):
    """Test on a single image with visual preview"""
    print(f"\n🔍 Analyzing: {image_path}")
    
    # Need dimensions first
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return
    h, w = img.shape[:2]
    
    detections = label_with_lm_studio(client, image_path, prompt)
    
    print(f"\n📦 Found {len(detections)} objects:")
    for det in detections:
        print(f"   - {det['class']}")
    
    for det in detections:
        raw_box = det.get('box', [0,0,0,0])
        x1, y1, x2, y2 = normalize_box(raw_box, w, h)
        
        # To pixels for drawing
        px1, py1 = int(x1*w), int(y1*h)
        px2, py2 = int(x2*w), int(y2*h)
        
        color = (0, 255, 0)
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
        cv2.putText(img, det['class'], (px1, py1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    preview_path = Path(image_path).parent / "lm_studio_preview.png"
    cv2.imwrite(str(preview_path), img)
    print(f"\n💾 Preview saved: {preview_path}")

def main():
    print("="*60)
    print("  LM STUDIO LOCAL LABELER")
    print(f"  Connecting to: {LM_STUDIO_URL}")
    print("="*60)
    
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    vocab_path = data_dir / "ui_vocabulary.json"
    dataset_dir = data_dir / "yolo_dataset"
    images_dir = dataset_dir / "images" / "train"
    
    # SEPARATE OUTPUT FOLDER
    labels_dir = data_dir / "lm_studio_labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    if not vocab_path.exists():
        print("No vocabulary found!")
        return

    # Init
    vocab = load_vocabulary(vocab_path)
    class_map = build_class_map(vocab)
    prompt = build_prompt_from_vocab(vocab)
    
    print(f"📚 Loaded {len(class_map)} classes")
    
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key=API_KEY)
        # Test connection
        client.models.list()
        print("✅ Connected to LM Studio")
    except Exception as e:
        print(f"❌ Could not connect to LM Studio at {LM_STUDIO_URL}")
        print("   Make sure the Local Server is RUNNING (Port 1234)")
        print(f"   Error: {e}")
        return
    
    # Options
    print("\n🎮 Options:")
    print("  1. Test on ONE image (with preview)")
    print("  2. Label ALL images")
    
    choice = input("\nChoice [1/2]: ").strip()
    
    images = list(images_dir.glob("*.png"))
    if not images:
        print("No images found in dataset!")
        return

    if choice == "1":
        # Pick random or first
        sample = images[0]
        preview_single_image(client, str(sample), prompt, class_map)
        
    else:
        # Check images
        pending = [img for img in images if not (labels_dir / f"{img.stem}.txt").exists()]
        
        if not pending:
            print("✅ All images labeled in lm_studio_labels!")
            return
            
        print(f"\n📸 Processing {len(pending)} images...")
        print(f"   Saving to: {labels_dir}")
        
        labeled_count = 0
        
        for i, img_path in enumerate(pending):
            print(f"   {i+1}/{len(pending)}: {img_path.name}...", end="", flush=True)
            
            try:
                # Need dimensions for normalization
                import PIL.Image
                with PIL.Image.open(img_path) as img:
                    w, h = img.size
                    
                detections = label_with_lm_studio(client, str(img_path), prompt)
                
                if detections:
                    yolo_lines = convert_to_yolo(detections, class_map, w, h)
                    
                    if yolo_lines:
                        label_path = labels_dir / f"{img_path.stem}.txt"
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_lines))
                        print(f" Found {len(yolo_lines)} objects")
                        labeled_count += 1
                    else:
                        print(" No match")
                else:
                    print(" Failed/Empty")
            except Exception as e:
                print(f" Error: {e}")
                
        print(f"\n✅ Done! Labeled {labeled_count} images locally.")

if __name__ == "__main__":
    main()
