"""
Dynamic Discovery Auto-Labeler

Self-improving YOLO system with Gemini:
- Triggers on low-confidence YOLO detections (0.3-0.6)
- Gemini classifies uncertain objects using a reference gallery
- Can discover NEW classes dynamically
- Saves full frames with YOLO-format labels
- Self-limiting: as YOLO improves, fewer Gemini calls

Usage:
    from training.auto_labeler import AutoLabeler
    labeler = AutoLabeler(output_dir="data/auto_labeled")
    labeler.start()
    # During gameplay:
    labeler.check_and_queue(frame, detections)
"""

import os
import time
import json
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2


class AutoLabeler:
    """
    Dynamic Discovery Auto-Labeler with Gemini integration.

    Architecture:
    1. YOLO detects objects with confidence scores
    2. Low-confidence (0.3-0.6) triggers Gemini classification
    3. Gemini sees: cropped object + reference gallery of known classes
    4. Gemini returns: class name (existing or NEW)
    5. Full frame saved with corrected/new labels
    6. As YOLO improves → fewer low-conf → fewer Gemini calls (self-limiting)
    """

    # Confidence thresholds
    LOW_CONF_MIN = 0.3   # Below this = noise, ignore
    LOW_CONF_MAX = 0.85  # Above this = YOLO is confident, trust it
    HIGH_CONF = 0.9      # Above this = definitely correct, add to gallery

    # Gallery settings
    MAX_GALLERY_PER_CLASS = 5  # Reference images per class
    MIN_CROP_SIZE = 32         # Minimum crop dimension

    def __init__(self,
                 output_dir: str = "data/auto_labeled",
                 max_queue_size: int = 20,
                 periodic_interval: int = 300,
                 gemini_api_key: str = None,
                 gemini_model: str = "gemini-2.0-flash",
                 # Legacy params (ignored)
                 device: str = None,
                 llm_url: str = None,
                 llm_model: str = None):
        """
        Args:
            output_dir: Where to save labeled data
            max_queue_size: Max frames in processing queue
            periodic_interval: Sample every N frames for periodic captures
            gemini_api_key: Google API key (or from GOOGLE_API env var)
            gemini_model: Gemini model to use
        """
        self.output_dir = Path(output_dir)
        self.periodic_interval = periodic_interval
        self.gemini_model = gemini_model

        # Get API key from param or environment
        self.api_key = gemini_api_key or os.environ.get('GOOGLE_API') or os.environ.get('GOOGLE_API_KEY')
        self.gemini_enabled = bool(self.api_key)

        # Create directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.gallery_dir = self.output_dir / "gallery"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

        # Threading
        self.frame_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Stats
        self.stats = {
            'frames_queued': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'labels_generated': 0,
            'gemini_calls': 0,
            'gemini_corrections': 0,
            'new_classes_discovered': 0,
            'high_conf_gallery_adds': 0,
            'start_time': None
        }

        # Frame counter
        self.frame_counter = 0

        # Class mapping: {class_name: class_id}
        self.class_map = {}
        self._load_class_map()

        # Reference gallery: {class_name: [crop_paths]}
        self.gallery = {}
        self._load_gallery()

        # Gemini client (lazy init)
        self._gemini = None
        self._gemini_client = None  # New google.genai client
        self._last_gemini_call = 0
        self.GEMINI_MIN_INTERVAL = 0.5  # Min seconds between Gemini calls (rate limiting)

    def _load_class_map(self):
        """Load existing class mapping."""
        path = self.output_dir / "classes.json"
        if path.exists():
            with open(path, 'r') as f:
                self.class_map = json.load(f)

    def _save_class_map(self):
        """Save class mapping."""
        path = self.output_dir / "classes.json"
        with open(path, 'w') as f:
            json.dump(self.class_map, f, indent=2)

    def set_class_map(self, class_map: Dict[int, str]):
        """Set class mapping from YOLO model."""
        # Convert {0: 'BonusBox', 1: 'Devo'} to {'BonusBox': 0, 'Devo': 1}
        self.class_map = {name: idx for idx, name in class_map.items()}
        self._save_class_map()

    def _load_gallery(self):
        """Load reference gallery from disk."""
        self.gallery = {}
        if not self.gallery_dir.exists():
            return
        for class_dir in self.gallery_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                crops = list(class_dir.glob("*.jpg"))
                if crops:
                    self.gallery[class_name] = [str(p) for p in crops[:self.MAX_GALLERY_PER_CLASS]]

    def _add_to_gallery(self, class_name: str, crop: np.ndarray) -> bool:
        """Add a high-confidence crop to the reference gallery."""
        if class_name not in self.gallery:
            self.gallery[class_name] = []

        if len(self.gallery[class_name]) >= self.MAX_GALLERY_PER_CLASS:
            return False  # Gallery full for this class

        # Create class directory
        class_dir = self.gallery_dir / class_name
        class_dir.mkdir(exist_ok=True)

        # Save crop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crop_path = class_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.gallery[class_name].append(str(crop_path))
        self.stats['high_conf_gallery_adds'] += 1
        return True

    def _get_gemini_client(self):
        """Lazy-initialize Gemini client using new google.genai SDK."""
        if self._gemini is None and self.api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
                self._gemini = self.gemini_model  # Store model name
                print(f"[AutoLabeler] Gemini initialized (new SDK): {self.gemini_model}")
            except ImportError:
                print("[AutoLabeler] google-genai not installed. Run: pip install google-genai")
                self.gemini_enabled = False
            except Exception as e:
                print(f"[AutoLabeler] Gemini init failed: {e}")
                self.gemini_enabled = False
        return self._gemini

    def _crop_detection(self, image: np.ndarray, bbox: List[float], padding: float = 0.1) -> Optional[np.ndarray]:
        """Crop detection from image with padding."""
        h, w = image.shape[:2]
        x1, y1, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]

        # Add padding
        pad_w = bw * padding
        pad_h = bh * padding

        # Convert to pixel coordinates with padding
        px1 = int(max(0, x1 - pad_w))
        py1 = int(max(0, y1 - pad_h))
        px2 = int(min(w, x1 + bw + pad_w))
        py2 = int(min(h, y1 + bh + pad_h))

        # Check minimum size
        if px2 - px1 < self.MIN_CROP_SIZE or py2 - py1 < self.MIN_CROP_SIZE:
            return None

        return image[py1:py2, px1:px2].copy()

    def _build_comparison_image(self, target_crop: np.ndarray, max_refs: int = 3) -> Tuple[np.ndarray, List[str]]:
        """
        Build a comparison image with:
        - Target object in RED FRAME (left side, larger)
        - Reference examples for each known class (right side, smaller)

        Layout:
        +----------------+------------------+
        |   TARGET       |  Class1: [ex][ex]|
        |   (red frame)  |  Class2: [ex][ex]|
        |                |  Class3: [ex][ex]|
        +----------------+------------------+
        """
        # Resize target to standard size
        target_size = 128
        target_resized = cv2.resize(target_crop, (target_size, target_size))

        # Add RED frame around target (3 pixels thick)
        cv2.rectangle(target_resized, (0, 0), (target_size-1, target_size-1), (0, 0, 255), 3)

        # Add "CLASSIFY THIS" label
        cv2.putText(target_resized, "CLASSIFY", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Get classes with gallery images
        classes_with_images = [(name, paths) for name, paths in self.gallery.items() if paths]

        if not classes_with_images:
            # No gallery yet - just return target with frame
            return target_resized, []

        # Build reference panel
        ref_cell_size = 48
        max_examples = min(max_refs, 3)

        # Calculate panel size
        panel_w = max_examples * ref_cell_size + 10  # +10 for padding
        panel_h = max(target_size, len(classes_with_images) * (ref_cell_size + 20))  # +20 for class label

        ref_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        ref_panel[:] = (30, 30, 30)  # Dark gray background

        class_names = []
        for row, (class_name, paths) in enumerate(classes_with_images[:8]):  # Max 8 classes
            class_names.append(class_name)
            y_start = row * (ref_cell_size + 20)

            # Draw class name
            cv2.putText(ref_panel, class_name[:12], (5, y_start + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw reference examples
            for col, path in enumerate(paths[:max_examples]):
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (ref_cell_size, ref_cell_size))
                        x = col * ref_cell_size + 5
                        y = y_start + 18
                        if y + ref_cell_size <= panel_h:
                            ref_panel[y:y+ref_cell_size, x:x+ref_cell_size] = img
                except:
                    pass

        # Combine target (left) and references (right)
        # Make sure heights match
        final_h = max(target_size, panel_h)
        final_w = target_size + panel_w + 10  # +10 gap

        combined = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        combined[:] = (20, 20, 20)  # Dark background

        # Place target on left
        combined[:target_size, :target_size] = target_resized

        # Place references on right
        combined[:panel_h, target_size+10:target_size+10+panel_w] = ref_panel

        return combined, class_names

    def _classify_with_gemini(self, crop: np.ndarray, yolo_class: str, yolo_conf: float) -> Tuple[str, float]:
        """
        Use Gemini to classify an uncertain detection.

        Sends a single comparison image with:
        - Target object in RED FRAME (left)
        - Reference examples of known classes (right)

        Returns:
            (class_name, confidence) - class_name may be NEW if Gemini discovers something
        """
        gemini = self._get_gemini_client()
        if not gemini:
            return yolo_class, yolo_conf  # Fallback to YOLO

        # Rate limiting
        now = time.time()
        time_since_last = now - self._last_gemini_call
        if time_since_last < self.GEMINI_MIN_INTERVAL:
            time.sleep(self.GEMINI_MIN_INTERVAL - time_since_last)

        try:
            from PIL import Image

            self._last_gemini_call = time.time()
            self.stats['gemini_calls'] += 1

            # Build comparison image: target with RED frame + reference gallery
            comparison_img, gallery_classes = self._build_comparison_image(crop)

            # Convert to PIL for Gemini
            comparison_pil = Image.fromarray(cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB))

            # Build prompt
            known_classes = list(self.class_map.keys())
            has_gallery = len(gallery_classes) > 0

            if has_gallery:
                prompt = f"""DarkOrbit game object classification.

IMAGE LAYOUT:
- LEFT (RED FRAME): Object to classify
- RIGHT: Reference examples of known classes

YOLO guess: "{yolo_class}" ({yolo_conf:.0%} confidence)

Known classes: {', '.join(known_classes)}

TASK: Classify the RED-FRAMED object on the left.
- If it matches a known class, return that exact name
- If YOLO is wrong, return the correct class
- If it's something NEW, give it a descriptive name

Respond with ONLY JSON:
{{"class": "ClassName", "confidence": 0.9, "reason": "why"}}

If unclear/noise/UI element:
{{"class": "IGNORE", "confidence": 0.0, "reason": "not a game object"}}"""
            else:
                prompt = f"""DarkOrbit game object classification.

The RED-FRAMED object needs to be classified.

YOLO guess: "{yolo_class}" ({yolo_conf:.0%} confidence)

Known classes: {', '.join(known_classes) if known_classes else 'None yet'}

TASK: Classify this object.
- If it matches a known class name, return that exact name
- If it's a game entity (alien, ship, box, etc.), name it appropriately
- If it's unclear/noise/UI, ignore it

Respond with ONLY JSON:
{{"class": "ClassName", "confidence": 0.9, "reason": "why"}}

If unclear/noise/UI element:
{{"class": "IGNORE", "confidence": 0.0, "reason": "not a game object"}}"""

            # Call Gemini with the single comparison image (new SDK)
            response = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[comparison_pil, prompt]
            )
            result_text = response.text.strip()

            # Parse JSON response (handle markdown code blocks)
            if "```" in result_text:
                # Extract JSON from code block
                parts = result_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        result_text = part
                        break

            result = json.loads(result_text)

            gemini_class = result.get('class', yolo_class)
            gemini_conf = result.get('confidence', 0.5)

            # Handle IGNORE
            if gemini_class == "IGNORE":
                return "IGNORE", 0.0

            # Check if this is a NEW class
            if gemini_class not in self.class_map:
                # Gemini discovered a new class!
                new_id = len(self.class_map)
                self.class_map[gemini_class] = new_id
                self._save_class_map()
                self.stats['new_classes_discovered'] += 1
                print(f"[AutoLabeler] 🆕 New class: {gemini_class} (id={new_id})")

            # Track corrections
            if gemini_class != yolo_class:
                self.stats['gemini_corrections'] += 1
                print(f"[AutoLabeler] Corrected: {yolo_class} → {gemini_class}")

            return gemini_class, gemini_conf

        except json.JSONDecodeError as e:
            print(f"[AutoLabeler] Gemini JSON parse error: {e}")
            return yolo_class, yolo_conf
        except Exception as e:
            print(f"[AutoLabeler] Gemini error: {e}")
            return yolo_class, yolo_conf

    def start(self):
        """Start the auto-labeler."""
        if self.running:
            return

        self.running = True
        self.stats['start_time'] = time.time()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        mode = "Dynamic Discovery (Gemini)" if self.gemini_enabled else "Simple (YOLO only)"
        print(f"[AutoLabeler] Running in {mode} mode")
        if self.gemini_enabled:
            print(f"   Low-conf trigger: {self.LOW_CONF_MIN:.0%}-{self.LOW_CONF_MAX:.0%}")
            print(f"   Known classes: {len(self.class_map)}")
            print(f"   Gallery images: {sum(len(v) for v in self.gallery.values())}")

    def stop(self):
        """Stop the auto-labeler."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)

        self._save_class_map()
        self._save_dataset_yaml()
        self._print_stats()

    def check_and_queue(self, image: np.ndarray, detections: List[Dict], force: bool = False) -> bool:
        """Check if frame should be queued for processing."""
        self.frame_counter += 1

        if self.frame_queue.full():
            return False

        # Force queue (user feedback)
        if force:
            return self.queue_frame(image, detections, "user_feedback")

        # Check for low-confidence detections that need Gemini
        if detections and self.gemini_enabled:
            uncertain = [d for d in detections
                        if self.LOW_CONF_MIN <= d.get('confidence', 1.0) <= self.LOW_CONF_MAX]
            if uncertain:
                return self.queue_frame(image, detections, "low_confidence")

        # Check for high-confidence detections to add to gallery
        if detections:
            high_conf = [d for d in detections if d.get('confidence', 0) >= self.HIGH_CONF]
            for det in high_conf:
                class_name = det.get('class_name', 'unknown')
                if class_name in self.class_map:
                    # Add to gallery if not full
                    current_count = len(self.gallery.get(class_name, []))
                    if current_count < self.MAX_GALLERY_PER_CLASS:
                        bbox = det.get('bbox', [0,0,0,0])
                        crop = self._crop_detection(image, bbox)
                        if crop is not None:
                            self._add_to_gallery(class_name, crop)

        # Periodic sampling
        if self.frame_counter % self.periodic_interval == 0:
            return self.queue_frame(image, detections, "periodic")

        return False

    def queue_frame(self, image: np.ndarray, detections: List[Dict] = None,
                   reason: str = "periodic", priority: int = 0) -> bool:
        """Queue a frame for processing."""
        if not self.running:
            return False

        if self.frame_queue.full():
            self.stats['frames_skipped'] += 1
            return False

        try:
            self.frame_queue.put_nowait({
                'image': image.copy(),
                'detections': detections or [],
                'reason': reason,
                'timestamp': time.time()
            })
            self.stats['frames_queued'] += 1
            return True
        except queue.Full:
            self.stats['frames_skipped'] += 1
            return False

    def _worker_loop(self):
        """Background worker that processes queued frames."""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_frame(frame_data)
                self.stats['frames_processed'] += 1
            except Exception as e:
                print(f"[AutoLabeler] Process error: {e}")
                import traceback
                traceback.print_exc()

    def _process_frame(self, frame_data: Dict):
        """Process a frame: classify uncertain detections, save labels."""
        image = frame_data['image']
        detections = frame_data['detections']
        reason = frame_data['reason']

        h, w = image.shape[:2]
        processed_detections = []

        for det in detections:
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.5)
            bbox = det.get('bbox', [0, 0, 0, 0])

            # Decide if we need Gemini classification
            needs_gemini = (self.gemini_enabled and
                          self.LOW_CONF_MIN <= confidence <= self.LOW_CONF_MAX)

            if needs_gemini:
                # Get crop for Gemini
                crop = self._crop_detection(image, bbox)
                if crop is not None:
                    # Ask Gemini to classify
                    new_class, new_conf = self._classify_with_gemini(crop, class_name, confidence)

                    if new_class == "IGNORE":
                        continue  # Skip this detection

                    class_name = new_class
                    confidence = new_conf

                    # Add to gallery if Gemini is confident
                    if new_conf >= self.HIGH_CONF:
                        self._add_to_gallery(class_name, crop)

            # Get or create class ID
            if class_name not in self.class_map:
                class_id = len(self.class_map)
                self.class_map[class_name] = class_id
            else:
                class_id = self.class_map[class_name]

            processed_detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox
            })

        # Save frame and labels
        self._save_frame(image, processed_detections, reason)

    def _save_frame(self, image: np.ndarray, detections: List[Dict], reason: str):
        """Save frame and YOLO-format labels."""
        h, w = image.shape[:2]

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{reason}_{timestamp}"

        # Save image
        img_path = self.images_dir / f"{base_name}.jpg"
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save labels (YOLO format: class_id cx cy w h)
        label_path = self.labels_dir / f"{base_name}.txt"
        label_count = 0

        with open(label_path, 'w') as f:
            for det in detections:
                class_id = det['class_id']
                bbox = det['bbox']

                if len(bbox) >= 4:
                    x1, y1, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Convert to normalized center coordinates
                    cx = (x1 + bw / 2) / w
                    cy = (y1 + bh / 2) / h
                    nw = bw / w
                    nh = bh / h

                    # Clamp to valid range
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))

                    if nw > 0.01 and nh > 0.01:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                        label_count += 1

        self.stats['labels_generated'] += label_count

    def _save_dataset_yaml(self):
        """Save dataset.yaml for YOLO training."""
        if not self.class_map:
            return

        yaml_path = self.output_dir / "dataset.yaml"

        # Create names list sorted by ID
        names = {v: k for k, v in self.class_map.items()}
        max_id = max(names.keys()) if names else -1
        names_list = [names.get(i, f"class_{i}") for i in range(max_id + 1)]

        content = f"""# Dynamic Discovery Dataset for YOLO Training
# Generated: {datetime.now().isoformat()}
# Classes discovered: {len(self.class_map)}

path: {self.output_dir.absolute()}
train: images
val: images

names:
"""
        for i, name in enumerate(names_list):
            content += f"  {i}: {name}\n"

        with open(yaml_path, 'w') as f:
            f.write(content)

    def _print_stats(self):
        """Print statistics."""
        duration = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0

        print("\n" + "="*60)
        print("  Auto-Labeler Statistics (Dynamic Discovery)")
        print("="*60)
        print(f"  Duration: {duration:.1f}s")
        print(f"  Frames processed: {self.stats['frames_processed']}")
        print(f"  Labels generated: {self.stats['labels_generated']}")
        print(f"  Gemini calls: {self.stats['gemini_calls']}")
        print(f"  Gemini corrections: {self.stats['gemini_corrections']}")
        print(f"  New classes discovered: {self.stats['new_classes_discovered']}")
        print(f"  Gallery additions: {self.stats['high_conf_gallery_adds']}")
        print(f"  Total classes: {len(self.class_map)}")
        print(f"  Output: {self.output_dir}")
        print("="*60)

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            **self.stats,
            'queue_size': self.frame_queue.qsize(),
            'total_classes': len(self.class_map),
            'gallery_size': sum(len(v) for v in self.gallery.values())
        }
