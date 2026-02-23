
print("DEBUG: Starting local_labeler.py...")
import os
import json
import time
from pathlib import Path
import torch
from PIL import Image
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    print("❌ Transformers not installed!")
    print("   Run: uv pip install transformers torch torchvision timm einops")
    exit(1)

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r') as f:
        return json.load(f)

def build_class_map(vocab):
    class_map = {}
    idx = 0
    # Game objects first
    for obj in vocab.get('classes', {}).get('game_objects', []):
        class_map[obj['name'].lower()] = idx
        idx += 1
    # Then UI
    for elem in vocab.get('classes', {}).get('ui_elements', []):
        class_map[elem['name'].lower()] = idx
        idx += 1
    return class_map

class LocalLabeler:
    def __init__(self):
        print("\n🚀 Loading Florence-2 Model (this may take a minute)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # smaller 'base' model for speed, 'large' for accuracy
        self.model_id = 'microsoft/Florence-2-large' # Upgraded for 3070 Ti
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        print(f"✅ Model loaded on {self.device.upper()}")

    def detect(self, image_path, text_prompts):
        """
        Use Phrase Grounding to find objects.
        Florence-2 logic: <CAPTION_TO_PHRASE_GROUNDING> + Text describing logical objects
        """
        image = Image.open(image_path).convert("RGB")
        
        # Construct a descriptive caption from prompts to find them
        # "Locate the enemy npc, space ship, bonus box..."
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = "Locate these items: " + ", ".join(text_prompts) + "."

        inputs = self.processor(text=task_prompt + text_input, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        # Parse result: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': ['label1', ...]}}
        return parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})

def convert_to_yolo(results, class_map, width, height):
    yolo_lines = []
    
    bboxes = results.get('bboxes', [])
    labels = results.get('labels', [])
    
    for bbox, label in zip(bboxes, labels):
        # Florence returns exact label from prompt, e.g. "enemy npc"
        # We need to match it to our class_map keys
        
        # Clean label key
        clean_label = label.lower().strip()
        
        # Try finding partial match if exact match fails
        class_id = class_map.get(clean_label)
        
        # If not found, try to map back (e.g. "bonus box" -> "bonus_box")
        if class_id is None:
            clean_label = clean_label.replace(" ", "_")
            class_id = class_map.get(clean_label)
        
        if class_id is None:
            continue
            
        x1, y1, x2, y2 = bbox
        
        # Florence returns absolute coords
        # YOLO needs center_x, center_y, w, h normalized
        
        # Normalize
        nx1, ny1, nx2, ny2 = x1/width, y1/height, x2/width, y2/height
        
        w = nx2 - nx1
        h = ny2 - ny1
        cx = nx1 + w/2
        cy = ny1 + h/2
        
        # Clamp
        cx, cy, w, h = max(0, cx), max(0, cy), max(0, w), max(0, h)
        cx, cy, w, h = min(1, cx), min(1, cy), min(1, w), min(1, h)
        
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
    return yolo_lines

def preview_single_image(labeler, image_path, prompts, class_map):
    import cv2
    import numpy as np
    
    print(f"\n🔍 Analyzing: {image_path}")
    results = labeler.detect(str(image_path), prompts)
    
    bboxes = results.get('bboxes', [])
    labels = results.get('labels', [])
    
    print(f"📦 Found {len(bboxes)} objects")
    
    img = cv2.imread(str(image_path))
    if img is None: return
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    preview_path = Path(image_path).parent / "florence_preview.png"
    cv2.imwrite(str(preview_path), img)
    print(f"💾 Preview saved: {preview_path}")

def main():
    print("="*60)
    print("  💻 FLORENCE-2 LOCAL LABELER (LARGE)")
    print("  High Accuracy Mode for 3070 Ti")
    print("="*60)
    
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    vocab_path = data_dir / "ui_vocabulary.json"
    dataset_dir = data_dir / "yolo_dataset"
    images_dir = dataset_dir / "images" / "train"
    
    # Save to separate folder
    labels_dir = data_dir / "florence_labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocab
    if not vocab_path.exists():
        print("No vocabulary found!")
        return
    vocab = load_vocabulary(vocab_path)
    class_map = build_class_map(vocab)
    print(f"📚 Loaded {len(class_map)} classes from vocabulary")
    
    # Florence prompts - we construct text from class names
    prompts = [k.replace("_", " ") for k in class_map.keys()]
    
    labeler = LocalLabeler()
    
    print("\n🎮 Options:")
    print("  1. Test on ONE image (with preview)")
    print("  2. Label ALL images")
    
    choice = input("\nChoice [1/2]: ").strip()
    
    images = list(images_dir.glob("*.png"))
    
    if choice == "1":
        if images:
            preview_single_image(labeler, images[0], prompts, class_map)
    else:
        pending = [img for img in images if not (labels_dir / f"{img.stem}.txt").exists()]
        
        if not pending:
            print("✅ All images labeled in florence_labels!")
            return
            
        print(f"\n📸 Processing {len(pending)} images...")
        
        labeled_count = 0
        for i, img_path in enumerate(pending):
            print(f"   {i+1}/{len(pending)}: {img_path.name}...", end="", flush=True)
            try:
                results = labeler.detect(str(img_path), prompts)
                with Image.open(img_path) as img:
                    w, h = img.size
                yolo_lines = convert_to_yolo(results, class_map, w, h)
                
                if yolo_lines:
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    print(f" Found {len(yolo_lines)} objects")
                    labeled_count += 1
                else:
                    print(" No match")
            except Exception as e:
                print(f" Error: {e}")
                
        print(f"\n✅ Done! Labeled {labeled_count} images.")
