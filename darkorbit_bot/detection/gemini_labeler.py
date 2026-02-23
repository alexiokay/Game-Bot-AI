"""
DarkOrbit Bot - Gemini 3 Flash Auto-Labeler
Uses Gemini 3 Flash (latest) for precise bounding box detection.
Returns coordinates in 0-1000 scale for YOLO training.

Usage:
    $env:GEMINI_API_KEY="your-key"
    python gemini_labeler.py
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import time


def load_vocabulary(vocab_path: str) -> dict:
    """Load vocabulary from JSON file"""
    with open(vocab_path, 'r') as f:
        return json.load(f)


def build_prompt_from_vocab(vocab: dict) -> str:
    """Build detection prompt from vocabulary"""
    prompt = """You are analyzing a screenshot from a space game (DarkOrbit/Space Aces).

BE EXTREMELY CONSERVATIVE - Only label objects you are 100% certain about!

CRITICAL RULES - MUST FOLLOW:
1. DO NOT label visual effects as objects!
   - Engine trails/exhaust (glowing lines/streaks behind ships) = NOT OBJECTS
   - Laser beams/shots (thin lines or streaks) = NOT OBJECTS
   - Explosions, sparks, particle effects = NOT OBJECTS
   - Glowing halos or auras around ships = NOT OBJECTS

2. DO NOT label background/static elements!
   - Space stations, bases, portals = IGNORE (unless portal/space_station/planet is in class list)
   - Background planets or asteroids = IGNORE
   - Distant structures = IGNORE
   - Focus ONLY on interactive gameplay objects (ships, cargo, NPCs)

3. Only label SOLID objects with clear, defined boundaries
4. If you're not 100% sure what something is, DO NOT label it
5. DRONES: Only label as "drone" if you see small, solid robotic ships orbiting a main ship
   - NOT glowing trails, NOT laser fire, NOT exhaust
   - Must be clearly separate solid objects near a ship

GAME OBJECTS (in the main game area):
"""
    for obj in vocab.get('classes', {}).get('game_objects', []):
        prompt += f"- {obj['name']}: {obj['description']}\n"
    
    prompt += "\nUI ELEMENTS:\n"
    for elem in vocab.get('classes', {}).get('ui_elements', []):
        prompt += f"- {elem['name']}: {elem['description']}\n"
    
    prompt += """
For EACH object you find, provide PRECISE bounding box coordinates.

COORDINATE SYSTEM (STRICT):
- Scale: EXACTLY 0 to 1000 (integer values only)
- 0 = top/left edge, 1000 = bottom/right edge
- Format: x_min, y_min, x_max, y_max
- DO NOT use 0-100 scale, DO NOT use 0-1 scale
- DO NOT guess coordinates - be precise

IMPORTANT: Find EVERY instance of each object type!
- If there are 3 group members, list ALL 3 group_member entries
- If there are 3 health bars, list ALL 3 group_health_bar entries
- If there are 5 drones, list ALL 5 drone entries
- If there are 4 ammo slots, list ALL 4 ammo_slot entries

Output ONLY a JSON array (NO markdown, NO explanation):
[
  {"class": "minimap_panel", "x_min": 100, "y_min": 50, "x_max": 300, "y_max": 350},
  {"class": "group_health_bar", "x_min": 850, "y_min": 200, "x_max": 950, "y_max": 220}
]

Rules:
- Coordinates MUST be 0-1000 scale (e.g., 850 not 85, not 0.85)
- Include EVERY instance of ALL visible objects and UI elements
- Use the EXACT class names from the list above
- DRONES are small solid ships near larger ships, NOT engine trails/exhaust
- Output ONLY JSON array, no markdown code blocks"""
    
    return prompt

    return prompt


def build_class_map(vocab: dict) -> dict:
    """Build class_id mapping from vocabulary"""
    class_map = {}
    idx = 0
    
    for obj in vocab.get('classes', {}).get('game_objects', []):
        class_map[obj['name']] = idx
        idx += 1
    
    for elem in vocab.get('classes', {}).get('ui_elements', []):
        class_map[elem['name']] = idx
        idx += 1
    
    return class_map


def update_dataset_yaml(dataset_dir: Path, class_map: dict):
    """Update dataset.yaml to match the vocabulary classes"""
    yaml_path = dataset_dir / "dataset.yaml"
    
    # Invert class map to get id -> name
    id_to_name = {v: k for k, v in class_map.items()}
    
    # Determine absolute path for dataset root (YOLO needs this)
    # Using relative paths for portability if possible, but YOLO often prefers absolute
    path_str = str(dataset_dir.absolute()).replace("\\", "/")
    
    yaml_content = f"""# DarkOrbit Dynamic Dataset
path: {path_str}
train: images/train
val: images/val

# Classes ({len(class_map)} total)
names:
"""
    for i in range(len(class_map)):
        name = id_to_name.get(i, f"class_{i}")
        yaml_content += f"  {i}: {name}\\n"
        
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"✅ Updated dataset.yaml with {len(class_map)} classes")


def load_golden_examples(golden_dir: Path, class_map: Dict, k: int = 3) -> List:
    """Load k random examples (image + yolo_txt) formatted for Gemini chat history"""
    import random
    
    if not golden_dir.exists():
        return []
        
    # Find pairs of png/jpg and txt
    images = list(golden_dir.glob("*.png")) + list(golden_dir.glob("*.jpg"))
    valid_pairs = []
    
    for img in images:
        txt = golden_dir / f"{img.stem}.txt"
        if txt.exists():
            valid_pairs.append((img, txt))
            
    if not valid_pairs:
        return []
        
    # Select k random
    selected = random.sample(valid_pairs, min(k, len(valid_pairs)))
    
    # Invert class map for decoding
    id_to_name = {v: k for k, v in class_map.items()}
    
    history_contents = []
    
    for img_path, txt_path in selected:
        # 1. Parse YOLO txt back to JSON format Gemini expects
        boxes = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                # YOLO is normalized center_x, center_y, w, h
                # We need to render this roughly for the model to understand, 
                # OR we just provide the JSON format we want it to output.
                # The model expects: {"class": "name", "box_2d": [MIN, MIN, MAX, MAX]} inside 0-1000 scale
                
                # But wait, our prompt asks for dictionary with keys like x_min...
                # Let's match the OUPUT format requested in the prompt.
                # Prompt asks for: [{"class": "...", "x_min": ..., ...}]
                
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx - w/2) * 1000)
                y1 = int((cy - h/2) * 1000)
                x2 = int((cx + w/2) * 1000)
                y2 = int((cy + h/2) * 1000)
                
                # Clamp
                x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
                
                cls_name = id_to_name.get(cls_id, "unknown")
                boxes.append({
                    "class": cls_name,
                    "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2
                })
        
        # 2. Construct User/Model pair
        # User: [Prompt, Image]
        # Model: JSON String
        
        # Note: In multi-turn, we don't repeat the full prompt every time usually, 
        # but for Few-Shot visual, it's safer to treat them as independent turns or just (Image -> JSON) pairs.
        # But Gemini API expects alternating user/model.
        
        import PIL.Image
        ex_img = PIL.Image.open(img_path)
        
        # Turn 1: User provides image (and implied prompt context)
        # We can just say "Label this." for examples to save tokens? 
        # Or best to simulate the exact task.
        history_contents.append({"role": "user", "parts": ["Label this image.", ex_img]})
        history_contents.append({"role": "model", "parts": [json.dumps(boxes)]})
        
    print(f"   ✨ Loaded {len(selected)} golden examples for context")
    return history_contents


def label_with_gemini(image_path: str, api_key: str, prompt: str, model_name: str = 'gemini-2.0-flash-exp', golden_dir: Path = None, class_map: Dict = None) -> List[Dict]:
    """Use Gemini to detect objects. Supports One-Shot/Few-Shot if golden_dir provided."""
    from google import genai
    import PIL.Image
    import time
    import random
    import json

    client = genai.Client(api_key=api_key)
    img = PIL.Image.open(image_path)
    
    # Helper to clean up content creation
    def _make_part(item):
        if isinstance(item, str):
            return {"text": item}
        elif isinstance(item, PIL.Image.Image):
            # Convert to bytes
            import io
            b = io.BytesIO()
            item.save(b, format="PNG")
            return {"inline_data": {"mime_type": "image/png", "data": b.getvalue()}}
        return item

    def _make_content(role, parts):
        return {"role": role, "parts": [_make_part(p) for p in parts]}

    # Load examples if available
    history = []
    if golden_dir and class_map:
        # Load examples locally but formatting logic needs to be here or helper needs to be shared
        # We will inline the logic of load_golden_examples here to access _make_content or redefine it
        
        # Let's just redefine a quick loader here that does it right
        import random
        golden_files = list(Path(golden_dir).glob("*.png")) + list(Path(golden_dir).glob("*.jpg"))
        valid_pairs = []
        for g_img in golden_files:
            g_txt = Path(golden_dir) / f"{g_img.stem}.txt"
            if g_txt.exists():
                valid_pairs.append((g_img, g_txt))
        
        selected = random.sample(valid_pairs, min(2, len(valid_pairs))) if valid_pairs else []
        
        # Invert class map
        id_to_name = {v: k for k, v in class_map.items()}
        
        for ex_img_path, ex_txt_path in selected:
            # Parse txt
            boxes = []
            with open(ex_txt_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = int((cx - w/2) * 1000)
                    y1 = int((cy - h/2) * 1000)
                    x2 = int((cx + w/2) * 1000)
                    y2 = int((cy + h/2) * 1000)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
                    cls_name = id_to_name.get(cls_id, "unknown")
                    boxes.append({"class": cls_name, "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2})
            
            ex_pil = PIL.Image.open(ex_img_path)
            history.append(_make_content("user", ["Label this image.", ex_pil]))
            history.append(_make_content("model", [json.dumps(boxes)]))
        
        if history:
             print(f"   ✨ Loaded {len(selected)} golden examples for context")


    # Retry loop for rate limits
    max_retries = 5
    base_wait = 5

    for attempt in range(max_retries):
        try:
            # Construct content
            # If we have history, we append the target image as the last user message
            
            chat_session = None
            
            if history:
                 # To use history with genai client, we might need a chat session or just raw contents list
                 # We'll use raw contents list [User, Model, User, Model, User(Target)]
                 
                 # Prepare the final turn
                 final_turn = {"role": "user", "parts": [prompt, img]}
                 
                 full_contents = history + [final_turn]
                 
                 # We need to be careful about the structure. 
                 # The prompt is large. We should probably put the System Prompt first, 
                 # then Examples (Image -> Json), then Target (Image).
                 
                 # Let's adjust:
                 # 1. System Instruction (The big prompt)
                 # 2. Few-shot examples
                 # 3. Target
                 
                 # genai.Client doesn't support system_instruction in generate_content directly the same way as chat? 
                 # Actually it does `config=...`
                 
                 # Let's simplify: 
                 # User: "Here is the rule set: [PROMPT]. Here are examples: [IMG1] -> [JSON1]. Now label this: [TARGET]"
                 # But sending multiple images in one turn is complex.
                 
                 # Strategy: Chat History
                 # User: [System Prompt] "Agree?"
                 # Model: "Yes." (Artificial)
                 # User: [Img1]
                 # Model: [Json1]
                 # User: [Target]
                 
                 pass # Logic below
            
            # Simple approach: If history exists, use it.
            if history:
                # We attach the SYSTEM PROMPT to the FIRST user message in history
                # history[0] is now a dict {"role": "user", "parts": [{"text": "Label...", ...}, ...]}
                
                # Careful: The first part might be text or image. We want to prepend to text.
                # Since we constructed it, we know parts[0] is "text": "Label this image."
                
                # Clone history to avoid modifying global state/cache if reused? 
                # (Though we construct it fresh each call currently)
                
                final_history = []
                # Deep copy basic structures
                import copy
                final_history = copy.deepcopy(history)
                
                first_text_part = final_history[0]['parts'][0]
                if 'text' in first_text_part:
                    first_text_part['text'] = prompt + "\n\nAnalyze this example image:\n" + first_text_part['text']
                else:
                    # Prepend a new text part if for some reason it wasn't there
                    final_history[0]['parts'].insert(0, {"text": prompt + "\n\nAnalyze this example image:"})
                
                # Add target
                # We need to use _make_content here too
                final_history.append(_make_content("user", ["Analyze this target image:", img]))
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=final_history,
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 8192
                    }
                )
            else:
                # Zero shot
                # We should also use explicit structure to be safe
                contents = [_make_content("user", [prompt, img])]
                # Or just pass the list if the top-level list works (it usually does), 
                # but let's be consistent.
                # Actually, genai client single-turn takes a list of parts or list of contents.
                # let's try strict list of Content objects (dicts)
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img], # Keep this simple legacy way if it worked before, or switch?
                    # "contents" argument in v1 client is flexible.
                    # But if we want to be 100% safe:
                    # contents=[_make_content("user", [prompt, img])],
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 8192
                    }
                )

            
            # Clean response text
            text = response.text.strip()

            # Remove markdown code blocks
            if text.startswith("```"):
                # Remove opening ```json or ```
                first_newline = text.find("\n")
                if first_newline != -1:
                    text = text[first_newline+1:]
                else:
                    text = text[3:]  # Just remove ```

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            # Remove any trailing commas before ] (common JSON error)
            import re
            text = re.sub(r',(\s*])', r'\1', text)
            text = re.sub(r',(\s*})', r'\1', text)

            # Fix incomplete JSON (model hit token limit mid-response)
            if not text.endswith(']'):
                # Remove incomplete last entry
                last_brace = text.rfind('{')
                if last_brace != -1:
                    text = text[:last_brace].rstrip()
                    # Remove trailing comma if present
                    if text.endswith(','):
                        text = text[:-1]
                # Close the array
                text += '\n]'
                print(f"   ⚠️  Fixed incomplete JSON (model output truncated)")

            # Parse JSON
            try:
                detections = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON Parse Error: {e}")
                print(f"Response length: {len(text)} chars")
                print(f"Response preview (first 300 chars):\n{text[:300]}")
                print(f"Response preview (last 200 chars):\n{text[-200:]}")

                # Save full response for debugging
                debug_path = Path(image_path).parent / "gemini_debug_response.txt"
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Full response saved to: {debug_path}")
                return []

            if not detections:
                return []

            # Validate and normalize coordinates (Gemini 3 should return 0-1000)
            all_coords = []
            for det in detections:
                all_coords.extend([
                    det.get('x_min', 0), det.get('y_min', 0),
                    det.get('x_max', 0), det.get('y_max', 0)
                ])

            max_val = max(all_coords) if all_coords else 0
            min_val = min(all_coords) if all_coords else 0

            # Determine scale intelligently
            if max_val <= 1.0:
                # Model returned 0-1 scale (wrong!)
                divisor = 1.0
                print(f"   ⚠️  WARNING: Model used 0-1 scale (expected 0-1000). Coords may be imprecise!")
            elif max_val <= 100:
                # Model returned 0-100 scale (wrong!)
                divisor = 100.0
                print(f"   ⚠️  WARNING: Model used 0-100 scale (expected 0-1000). Coords may be imprecise!")
            else:
                # Correct: 0-1000 scale
                divisor = 1000.0
                print(f"   ✅ Using 0-1000 coordinate scale (max: {max_val:.0f}, min: {min_val:.0f})")

            # Normalize to 0-1 for YOLO format
            normalized_detections = []
            for det in detections:
                x_min = det.pop('x_min', 0) / divisor
                y_min = det.pop('y_min', 0) / divisor
                x_max = det.pop('x_max', 0) / divisor
                y_max = det.pop('y_max', 0) / divisor

                # Validate bounding box (must be valid rectangle)
                if not (x_max > x_min and y_max > y_min and all(0 <= c <= 1 for c in [x_min, y_min, x_max, y_max])):
                    print(f"   ⚠️  Skipping invalid box for {det.get('class', 'unknown')}: [{x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f}]")
                    continue

                # Size validation for drones (reject if too large - likely visual effects)
                class_name = det.get('class', '').lower()
                width = x_max - x_min
                height = y_max - y_min

                if class_name == 'drone':
                    # Drones should be small (< 5% of screen width/height)
                    if width > 0.05 or height > 0.05:
                        print(f"   ⚠️  Skipping oversized 'drone' (likely visual effect): size {width*100:.1f}% x {height*100:.1f}%")
                        continue

                det['box'] = [x_min, y_min, x_max, y_max]
                normalized_detections.append(det)

            return normalized_detections
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_wait * (attempt + 1) + (random.randint(0, 1000) / 1000)
                print(f"   ⏳ Rate limit (429), waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                # Other errors are likely parsing or API errors we can't retry
                print(f"\n❌ API/Parse Error on {Path(image_path).name}: {e}")
                # print(f"Response: {response.text[:200] if 'response' in locals() and response.text else 'empty'}")
                return []
                
    return []


def convert_to_yolo(detections: List[Dict], class_map: Dict[str, int]) -> List[str]:
    """Convert to YOLO label format"""
    lines = []
    
    for det in detections:
        class_name = det.get('class', '').lower()
        if class_name not in class_map:
            continue
        
        class_id = class_map[class_name]
        box = det.get('box', [0, 0, 0, 0])
        
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        if width > 0 and height > 0:
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return lines


def preview_single_image(image_path: str, api_key: str, prompt: str):
    """Test on a single image with visual preview"""
    print(f"\n🔍 Analyzing: {image_path}")
    
    detections = label_with_gemini(image_path, api_key, prompt)
    
    print(f"\n📦 Found {len(detections)} objects:")
    for det in detections:
        print(f"   - {det['class']}")
    
    # Create visual preview
    try:
        import cv2
        import random
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Generate consistent colors for each class
        color_cache = {}
        def get_color(class_name):
            if class_name not in color_cache:
                random.seed(hash(class_name))
                color_cache[class_name] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
            return color_cache[class_name]
        
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
            color = get_color(det['class'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, det['class'], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        preview_path = Path(image_path).parent / "gemini_preview.png"
        cv2.imwrite(str(preview_path), img)
        print(f"\n💾 Preview saved: {preview_path}")
        
    except Exception as e:
        print(f"Preview error: {e}")
    
    return detections



def auto_label_all(images_dir: str, labels_dir: str, api_key: str, 
                   max_images: int = None, delay: float = 0.5,
                   prompt: str = None, class_map: dict = None, max_workers: int = 5, 
                   golden_dir: Path = None, model_name: str = 'gemini-2.0-flash-exp', force: bool = False):
    """Label all images using vocabulary with PARALLEL PROCESSING"""
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    
    if max_images:
        images = images[:max_images]
    
    # Check which ones are already labeled
    pending_images = []
    for img in images:
        if force or not (labels_path / f"{img.stem}.txt").exists():
            pending_images.append(img)
            
    if not pending_images:
        print(f"\n✅ All {len(images)} images already labeled!")
        return

    print(f"\n📸 Processing {len(pending_images)} images (parallel, {max_workers} threads)...")
    print(f"   Using {len(class_map)} classes from vocabulary")
    
    import concurrent.futures
    import threading
    
    labeled_count = 0
    total_objects = 0
    lock = threading.Lock()
    
    def process_image(img_path):
        nonlocal labeled_count, total_objects
        try:
            detections = label_with_gemini(str(img_path), api_key, prompt, model_name=model_name, golden_dir=golden_dir, class_map=class_map)

            
            if detections:
                yolo_lines = convert_to_yolo(detections, class_map)
                
                if yolo_lines:
                    label_path = labels_path / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    
                    with lock:
                        labeled_count += 1
                        total_objects += len(yolo_lines)
                        print(f"\r   Labeled: {labeled_count}/{len(pending_images)} | Found: {len(yolo_lines)} objs | {img_path.name}", end="", flush=True)
            
            # Rate limiting sleep
            time.sleep(delay)
            
        except Exception as e:
            # Print error to help debug
            with lock:
                print(f"\n❌ Error on {img_path.name}: {e}")
            pass
            
    # Run in parallel
    print(f"   Starting {max_workers} worker threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_image, pending_images)
    
    print(f"\n\n✅ Done!")
    print(f"   Newly labeled: {labeled_count}/{len(pending_images)}")
    print(f"   Total objects: {total_objects}")
    

def rename_files_in_dir(directory: Path):
    """Rename files to standard format img_timestamp_hash.png"""
    import hashlib
    import shutil
    
    print(f"\n🔄 Renaming files in {directory}...")
    files = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
    count = 0
    
    for f in files:
        if f.name.startswith("img_") and len(f.name) > 15:
            continue # Already renamed likely
            
        # Create unique name based on content hash to avoid duplicates
        with open(f, "rb") as bf:
            file_hash = hashlib.md5(bf.read()).hexdigest()[:8]
        
        timestamp = int(time.time())
        new_name = f"img_{timestamp}_{file_hash}{f.suffix}"
        new_path = directory / new_name
        
        if not new_path.exists():
            f.rename(new_path)
            count += 1
            
    print(f"   Renamed {count} files.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemini Auto-Labeler')
    parser.add_argument('--dir', type=str, help='Custom input directory (e.g. data/images/train_manual)')
    parser.add_argument('--rename', action='store_true', help='Auto-rename files to pattern img_TIMESTAMP_HASH')
    parser.add_argument('--golden', type=str, help='Directory of perfect examples for Few-Shot learning')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash-exp', help='Gemini model name (e.g. gemini-1.5-pro)')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel threads')
    parser.add_argument('--force', action='store_true', help='Overwrite existing labels')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"  🤖 GEMINI AUTO-LABELER | Model: {args.model}")
    print("="*60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not set!")
        api_key = input("Enter API Key: ").strip()
        if not api_key:
            return
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    dataset_dir = data_dir / "yolo_dataset"
    vocab_path = data_dir / "ui_vocabulary.json"
    
    golden_dir = None
    if args.golden:
        golden_dir = Path(args.golden)
        print(f"✨ Active Learning: Using golden examples from {golden_dir}")

    # Logic for custom directory
    if args.dir:
        train_images = Path(args.dir)
        # Try to infer labels dir: if images/foo -> labels/foo
        if "images" in str(train_images):
            labels_part = str(train_images).replace("images", "labels")
            train_labels = Path(labels_part)
        else:
            # Fallback
            train_labels = train_images.parent / (train_images.name + "_labels")
            
        print(f"\n📂 Using custom image dir: {train_images}")
        print(f"📂 Output labels dir: {train_labels}")
    else:
        train_images = dataset_dir / "images" / "train"
        train_labels = dataset_dir / "labels" / "train"

    
    if not train_images.exists():
        print(f"\n❌ Directory not found: {train_images}")
        return

    # Auto-rename if requested
    if args.rename:
        rename_files_in_dir(train_images)
    
    # Load vocabulary
    if vocab_path.exists():
        vocab = load_vocabulary(str(vocab_path))
        prompt = build_prompt_from_vocab(vocab)
        class_map = build_class_map(vocab)
        print(f"\n📚 Loaded vocabulary: {len(class_map)} classes")
    else:
        print(f"\n⚠️  No vocabulary found at {vocab_path}")
        return
    
    if args.dir:
        # Direct run if CLI args used
        auto_label_all(str(train_images), str(train_labels), api_key, prompt=prompt, class_map=class_map, max_workers=args.workers, golden_dir=golden_dir, model_name=args.model, force=args.force)
        return

    # Interactive mode (Legacy)
    print("\n🎮 Options:")
    print("  1. Test on ONE image")
    print("  2. Label first 10 images")
    print("  3. Label first 50 images")
    print("  4. Label ALL images")
    print("  5. Label Custom Folder...")
    
    choice = input("\nChoice [1-5]: ").strip() or "1"
    
    files = list(train_images.glob("*.png")) + list(train_images.glob("*.jpg"))
    if not files:
        print("No images found.")
        return
    sample = files[0]
    
    if choice == "1":
        preview_single_image(str(sample), api_key, prompt)
    elif choice == "2":
        auto_label_all(str(train_images), str(train_labels), api_key, 10, prompt=prompt, class_map=class_map, golden_dir=golden_dir, model_name=args.model, force=args.force)
    elif choice == "3":
        auto_label_all(str(train_images), str(train_labels), api_key, 50, prompt=prompt, class_map=class_map, golden_dir=golden_dir, model_name=args.model, force=args.force)
    elif choice == "4":
        threads_input = input("Enter number of threads [default=5]: ").strip()
        max_workers = int(threads_input) if threads_input.isdigit() else 5
        auto_label_all(str(train_images), str(train_labels), api_key, prompt=prompt, class_map=class_map, max_workers=max_workers, golden_dir=golden_dir, model_name=args.model, force=args.force)
    elif choice == "5":
        custom_path = input("Enter absolute path to folder: ").strip()
        custom_dir = Path(custom_path)
        if custom_dir.exists():
             # Try to infer output
            if "images" in str(custom_dir):
                out_dir = Path(str(custom_dir).replace("images", "labels"))
            else:
                out_dir = custom_dir.parent / (custom_dir.name + "_labels")
            
            should_rename = input("Rename files to standard format? (y/n): ").lower() == 'y'
            if should_rename:
                rename_files_in_dir(custom_dir)
                
            auto_label_all(str(custom_dir), str(out_dir), api_key, prompt=prompt, class_map=class_map, golden_dir=golden_dir, model_name=args.model, force=args.force)

            
            print("\n🖌️  Visualize results?")
            viz = input("Draw boxes on images? (y/n): ").strip().lower()
            if viz == 'y':
                import visualize_labels
                # infer output of vis
                vis_out = custom_dir / "preview_boxes"
                print(f"   Generating previews in {vis_out}...")
                
                # We need to load names map
                id_to_name = {v: k for k, v in class_map.items()}
                visualize_labels.visualize(custom_dir, out_dir, vis_out, id_to_name)


if __name__ == "__main__":
    main()
