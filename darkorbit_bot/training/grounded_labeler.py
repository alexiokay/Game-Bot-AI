#!/usr/bin/env python
"""
Grounded SAM 3 + Florence-2 Auto-Labeler

Uses text-guided detection instead of blind auto-segmentation:
1. Florence-2 or Grounding DINO finds objects via text prompts
2. SAM3 refines bounding boxes into precise polygon masks
3. Saves as YOLO segmentation training data (polygons, not just boxes)

This approach is cleaner than blind SAM auto-segmentation because:
- Only detects what you ASK for (no random text/artifacts)
- Zero-shot = no training needed, just text prompts
- You control the class list via ontology
- Much cleaner output
- Polygon masks for better segmentation training

Usage:
    python run_grounded.py --manual
    python run_grounded.py --auto --interval 60
"""

import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2


class GroundedLabeler:
    """
    Auto-labeler using Grounded SAM 3 (Florence-2 + SAM3).

    Instead of segmenting everything blindly, this uses text prompts
    to find specific objects, resulting in much cleaner detections.
    Outputs polygon masks in YOLO segmentation format.
    """

    # Polygon settings
    USE_POLYGON_MASKS = True
    POLYGON_SIMPLIFY_EPSILON = 2.0  # cv2.approxPolyDP epsilon for simplification

    # DarkOrbit ontology: text prompt -> class name
    # Format: "visual description for detection" -> "YOLOClassName"
    # NOTE: Grounding DINO works better with SHORT, SIMPLE prompts
    DEFAULT_ONTOLOGY = {
        # Game objects in space - use simple nouns
        "spaceship": "PlayerShip",
        "alien": "Streuner",  # Generic - will catch most aliens
        "orange alien": "Lordakia",
        "green alien": "Saimon",
        "crystal": "Kristallin",
        "box": "BonusBox",
        "cargo": "CargoBox",
        "portal": "Portal",
        "planet": "Planet",
        "drone": "Drone",
        "enemy ship": "EnemyPlayer",

        # UI elements - simple terms
        "minimap": "Minimap",
        "window": "ShipWindow",
        "chat": "ChatWindow",
        "menu": "MenuBar",
        "toolbar": "Hotbar",
    }

    def __init__(self,
                 output_dir: str = "data/grounded",
                 ontology: Dict[str, str] = None,
                 use_florence: bool = True,  # True = Florence-2, False = Grounding DINO
                 confidence_threshold: float = 0.3,
                 use_sam2_refinement: bool = True,  # Kept for backwards compat, enables SAM3
                 use_sam3_refinement: bool = True):
        """
        Initialize the grounded labeler.

        Args:
            output_dir: Where to save training data
            ontology: Dict mapping text prompts to class names
            use_florence: Use Florence-2 (True) or Grounding DINO (False)
            confidence_threshold: Minimum detection confidence
            use_sam2_refinement: Backwards compat - enables SAM refinement (uses SAM3)
            use_sam3_refinement: Whether to refine boxes with SAM3 masks (polygon output)
        """
        self.output_dir = Path(output_dir)
        self.ontology = ontology or self.DEFAULT_ONTOLOGY
        self.use_florence = use_florence
        self.confidence_threshold = confidence_threshold
        # SAM3 is used by default, fall back to SAM2 if not available
        self.use_sam_refinement = use_sam3_refinement or use_sam2_refinement

        # Create directories
        self.images_dir = self.output_dir / "images" / "train"
        self.labels_dir = self.output_dir / "labels" / "train"
        self.gallery_dir = self.output_dir / "gallery"
        self.debug_dir = self.output_dir / "debug"

        for d in [self.images_dir, self.labels_dir, self.gallery_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Build class map from ontology
        self.class_map = {}
        for i, class_name in enumerate(sorted(set(self.ontology.values()))):
            self.class_map[class_name] = i
        self._save_class_map()
        self._save_dataset_yaml()

        # Lazy-loaded models
        self._florence = None
        self._florence_processor = None
        self._grounding_dino = None
        self._sam3 = None  # SAM3 model (Ultralytics SAM)
        self._sam3_predictor = None
        self._sam2_predictor = None  # Fallback

        # Processing queue and thread
        self.frame_queue = queue.Queue(maxsize=5)
        self._running = False
        self._worker_thread = None

        # Stats
        self.stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'objects_labeled': 0,
        }

        print(f"[Grounded] Initialized with {len(self.ontology)} prompts -> {len(self.class_map)} classes")
        print(f"[Grounded] Detector: {'Florence-2' if use_florence else 'Grounding DINO'}")
        print(f"[Grounded] SAM3 refinement: {self.use_sam_refinement}")
        print(f"[Grounded] Polygon masks: {self.USE_POLYGON_MASKS}")

    def _save_class_map(self):
        """Save class mapping to JSON."""
        path = self.output_dir / "classes.json"
        with open(path, 'w') as f:
            json.dump(self.class_map, f, indent=2)

    def _save_dataset_yaml(self):
        """Save YOLO dataset.yaml."""
        yaml_content = f"""# DarkOrbit Grounded SAM Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/train

names:
"""
        for name, idx in sorted(self.class_map.items(), key=lambda x: x[1]):
            yaml_content += f"  {idx}: {name}\n"

        with open(self.output_dir / "dataset.yaml", 'w') as f:
            f.write(yaml_content)

    def _get_florence(self):
        """Lazy-load Florence-2 model."""
        if self._florence is None:
            try:
                import torch
                from transformers import AutoProcessor, AutoModelForCausalLM

                print("[Grounded] Loading Florence-2...")
                model_id = "microsoft/Florence-2-base"

                self._florence = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                self._florence_processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )

                if torch.cuda.is_available():
                    self._florence = self._florence.cuda()

                print(f"[Grounded] Florence-2 loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}")

            except ImportError as e:
                print(f"[Grounded] Florence-2 not available: {e}")
                print("[Grounded] Install: pip install transformers torch")
                return None, None
            except Exception as e:
                print(f"[Grounded] Error loading Florence-2: {e}")
                return None, None

        return self._florence, self._florence_processor

    def _get_grounding_dino(self):
        """Lazy-load Grounding DINO model."""
        if self._grounding_dino is None:
            try:
                import torch
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

                print("[Grounded] Loading Grounding DINO...")
                model_id = "IDEA-Research/grounding-dino-tiny"

                self._grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self._grounding_dino_processor = AutoProcessor.from_pretrained(model_id)

                if torch.cuda.is_available():
                    self._grounding_dino = self._grounding_dino.cuda()

                print(f"[Grounded] Grounding DINO loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}")

            except ImportError as e:
                print(f"[Grounded] Grounding DINO not available: {e}")
                print("[Grounded] Install: pip install transformers torch")
                return None
            except Exception as e:
                print(f"[Grounded] Error loading Grounding DINO: {e}")
                return None

        return self._grounding_dino

    def _get_sam3(self):
        """Lazy-load SAM3 model via Ultralytics."""
        if self._sam3 is None:
            try:
                from ultralytics import SAM

                # SAM3 model paths to try
                sam3_paths = [
                    "sam3.pt",
                    "models/sam3.pt",
                    "F:/dev/bot/models/sam3.pt",
                    "sam2.1_hiera_small.pt",  # Fallback to SAM2 via ultralytics
                    "models/sam2.1_hiera_small.pt",
                    "F:/dev/bot/models/sam2.1_hiera_small.pt",
                ]

                model_path = None
                for p in sam3_paths:
                    if Path(p).exists():
                        model_path = p
                        break

                if model_path:
                    print(f"[Grounded] Loading SAM: {model_path}")
                    self._sam3 = SAM(model_path)
                else:
                    # Try SAM2 models first (smaller, faster to download)
                    sam2_models = ["sam2_s.pt", "sam2_b.pt", "mobile_sam.pt", "sam_b.pt"]
                    loaded = False
                    for sam_model in sam2_models:
                        try:
                            print(f"[Grounded] Downloading SAM model: {sam_model}...")
                            self._sam3 = SAM(sam_model)
                            loaded = True
                            break
                        except Exception as e:
                            print(f"[Grounded] Failed to load {sam_model}: {e}")
                            continue

                    if not loaded:
                        print("[Grounded] All SAM models failed, trying sam_l.pt (large, ~2.4GB)...")
                        self._sam3 = SAM("sam_l.pt")

                print(f"[Grounded] SAM loaded successfully")

            except Exception as e:
                print(f"[Grounded] SAM3 not available: {e}")
                print("[Grounded] Trying SAM2 fallback...")
                return self._get_sam2_predictor_fallback()

        return self._sam3

    def _get_sam2_predictor_fallback(self):
        """Fallback to SAM2 if SAM3 not available."""
        if self._sam2_predictor is None:
            try:
                import torch
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                # Find SAM2 checkpoint
                checkpoint_paths = [
                    "models/sam2.1_hiera_small.pt",
                    "F:/dev/bot/models/sam2.1_hiera_small.pt",
                    "sam2.1_hiera_small.pt",
                ]

                checkpoint_path = None
                for p in checkpoint_paths:
                    if Path(p).exists():
                        checkpoint_path = p
                        break

                if not checkpoint_path:
                    print("[Grounded] SAM2 checkpoint not found, skipping refinement")
                    return None

                print(f"[Grounded] Loading SAM2 fallback: {checkpoint_path}")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam2_model = build_sam2("sam2.1_hiera_s", checkpoint_path, device=device)
                self._sam2_predictor = SAM2ImagePredictor(sam2_model)

                print(f"[Grounded] SAM2 predictor loaded on {device}")

            except ImportError as e:
                print(f"[Grounded] SAM2 fallback not available: {e}")
                return None
            except Exception as e:
                print(f"[Grounded] Error loading SAM2 fallback: {e}")
                return None

        return self._sam2_predictor

    def _detect_with_florence(self, image: np.ndarray, prompt: str) -> List[Dict]:
        """
        Detect objects using Florence-2 with a text prompt.

        Returns list of detections: [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'label': str}]
        """
        import torch
        from PIL import Image

        model, processor = self._get_florence()
        if model is None:
            return []

        # Convert to PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)

        # Use phrase grounding task
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        full_prompt = task + prompt

        inputs = processor(text=full_prompt, images=pil_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height)
        )

        detections = []
        if task in parsed and 'bboxes' in parsed[task]:
            bboxes = parsed[task]['bboxes']
            labels = parsed[task].get('labels', [prompt] * len(bboxes))

            for bbox, label in zip(bboxes, labels):
                # Florence returns [x1, y1, x2, y2]
                detections.append({
                    'bbox': bbox,
                    'confidence': 0.8,  # Florence doesn't return confidence
                    'label': label
                })

        return detections

    def _detect_with_grounding_dino(self, image: np.ndarray, prompt: str) -> List[Dict]:
        """
        Detect objects using Grounding DINO with a text prompt.

        Returns list of detections: [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'label': str}]
        """
        import torch
        from PIL import Image

        model = self._get_grounding_dino()
        if model is None:
            return []

        # Convert to PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)

        inputs = self._grounding_dino_processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process - API varies by transformers version
        try:
            # Newer API (transformers >= 4.36)
            results = self._grounding_dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]
            )[0]
        except TypeError:
            # Older API fallback
            results = self._grounding_dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=[pil_image.size[::-1]]
            )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            # Filter by confidence threshold manually if needed
            if score.item() >= self.confidence_threshold:
                detections.append({
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': score.item(),
                    'label': label
                })

        return detections

    def _refine_with_sam(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Refine a bounding box into a precise mask using SAM3 (or SAM2 fallback).

        Returns the mask or None if refinement fails.
        """
        if not self.use_sam_refinement:
            return None

        # Try SAM3 first
        sam3 = self._get_sam3()
        if sam3 is not None:
            return self._refine_with_sam3(image, bbox, sam3)

        # Fallback to SAM2 predictor
        predictor = self._get_sam2_predictor_fallback()
        if predictor is not None:
            return self._refine_with_sam2_predictor(image, bbox, predictor)

        return None

    def _refine_with_sam3(self, image: np.ndarray, bbox: List[float], sam3) -> Optional[np.ndarray]:
        """Refine using SAM3 via Ultralytics."""
        try:
            # SAM3 via ultralytics expects BGR image
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Run SAM3 with bounding box prompt
            results = sam3(image, bboxes=[[x1, y1, x2, y2]])

            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    return masks[0].astype(bool)

        except Exception as e:
            print(f"[Grounded] SAM3 refinement error: {e}")

        return None

    def _refine_with_sam2_predictor(self, image: np.ndarray, bbox: List[float], predictor) -> Optional[np.ndarray]:
        """Refine using SAM2 predictor (fallback)."""
        try:
            # Set image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            predictor.set_image(image_rgb)

            # Convert bbox to numpy array [x1, y1, x2, y2]
            box = np.array(bbox)

            # Predict mask from box
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=False
            )

            if len(masks) > 0:
                return masks[0]

        except Exception as e:
            print(f"[Grounded] SAM2 refinement error: {e}")

        return None

    def _mask_to_polygon(self, mask: np.ndarray, simplify: bool = True) -> List[float]:
        """
        Convert binary mask to polygon coordinates for YOLO segmentation format.

        Returns list of normalized [x1, y1, x2, y2, ...] coordinates.
        """
        h, w = mask.shape[:2]

        # Ensure mask is uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify polygon if requested
        if simplify:
            contour = cv2.approxPolyDP(contour, self.POLYGON_SIMPLIFY_EPSILON, True)

        # Need at least 3 points for a valid polygon
        if len(contour) < 3:
            return []

        # Convert to normalized [x1, y1, x2, y2, ...] format
        polygon = []
        for point in contour:
            px, py = point[0]
            polygon.extend([px / w, py / h])

        return polygon

    def detect_all(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all objects defined in ontology.

        Returns list of detections with class names.
        """
        all_detections = []

        # Group prompts by class for efficiency
        for prompt, class_name in self.ontology.items():
            if self.use_florence:
                dets = self._detect_with_florence(image, prompt)
            else:
                dets = self._detect_with_grounding_dino(image, prompt)

            for det in dets:
                if det['confidence'] >= self.confidence_threshold:
                    det['class_name'] = class_name
                    det['prompt'] = prompt
                    all_detections.append(det)

        # Optional: refine with SAM3 (or SAM2 fallback)
        if self.use_sam_refinement:
            for det in all_detections:
                mask = self._refine_with_sam(image, det['bbox'])
                if mask is not None:
                    # Update bbox from mask
                    ys, xs = np.where(mask)
                    if len(xs) > 0 and len(ys) > 0:
                        det['bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        det['mask'] = mask

        # Remove duplicates (same class, overlapping boxes)
        all_detections = self._remove_duplicates(all_detections)

        return all_detections

    def _remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate detections of the same class with overlapping boxes."""
        if len(detections) <= 1:
            return detections

        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class_name']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)

        # NMS per class
        result = []
        for cls, dets in by_class.items():
            # Sort by confidence
            dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)

            keep = []
            for det in dets:
                should_keep = True
                for kept in keep:
                    iou = self._compute_iou(det['bbox'], kept['bbox'])
                    if iou > iou_threshold:
                        should_keep = False
                        break
                if should_keep:
                    keep.append(det)

            result.extend(keep)

        return result

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _save_training_sample(self, image: np.ndarray, detections: List[Dict]):
        """Save image and YOLO-format labels (polygon segmentation or bounding boxes)."""
        if not detections:
            return

        h, w = image.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Save image
        img_path = self.images_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save labels in YOLO format
        label_path = self.labels_dir / f"{timestamp}.txt"
        polygon_count = 0
        bbox_count = 0

        with open(label_path, 'w') as f:
            for det in detections:
                class_name = det['class_name']
                if class_name not in self.class_map:
                    continue

                class_id = self.class_map[class_name]

                # Try polygon mask first (YOLO segmentation format)
                if self.USE_POLYGON_MASKS and 'mask' in det and det['mask'] is not None:
                    polygon = self._mask_to_polygon(det['mask'])
                    if polygon and len(polygon) >= 6:  # At least 3 points (6 coords)
                        coords_str = ' '.join(f"{c:.6f}" for c in polygon)
                        f.write(f"{class_id} {coords_str}\n")
                        polygon_count += 1
                        continue

                # Fallback to bounding box (YOLO detection format)
                x1, y1, x2, y2 = det['bbox']
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                bbox_count += 1

        self.stats['objects_labeled'] += len(detections)
        print(f"[Grounded] Saved {len(detections)} detections ({polygon_count} polygons, {bbox_count} boxes) to {img_path.name}")

    def _save_debug_image(self, image: np.ndarray, detections: List[Dict]):
        """Save debug visualization with masks and bounding boxes."""
        debug_img = image.copy()

        # Color palette for different classes
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            class_name = det['class_name']
            conf = det['confidence']
            color = colors[i % len(colors)]

            # Draw mask overlay if available
            if 'mask' in det and det['mask'] is not None:
                mask = det['mask']
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8)
                # Create colored overlay
                overlay = debug_img.copy()
                overlay[mask > 0] = color
                cv2.addWeighted(overlay, 0.3, debug_img, 0.7, 0, debug_img)

                # Draw mask contour
                contours, _ = cv2.findContours(
                    (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask,
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(debug_img, contours, -1, color, 2)

            # Draw bounding box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(debug_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_path = self.debug_dir / f"{timestamp}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)

    def process_frame(self, image: np.ndarray) -> List[Dict]:
        """Process a single frame and save training data."""
        self.stats['frames_processed'] += 1

        print(f"[Grounded] Processing frame {self.stats['frames_processed']}...")

        detections = self.detect_all(image)
        self.stats['detections_found'] += len(detections)

        if detections:
            self._save_training_sample(image, detections)
            self._save_debug_image(image, detections)
            print(f"[Grounded] Found {len(detections)} objects: {[d['class_name'] for d in detections]}")
        else:
            print("[Grounded] No objects detected")

        return detections

    def _worker_loop(self):
        """Background worker for processing frames."""
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                self.process_frame(frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Grounded] Worker error: {e}")
                import traceback
                traceback.print_exc()

    def start(self):
        """Start background processing thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print("[Grounded] Background worker started")

    def stop(self):
        """Stop background processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        print(f"[Grounded] Stopped. Stats: {self.stats}")

    def queue_frame(self, frame: np.ndarray):
        """Queue a frame for processing."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            print("[Grounded] Queue full, skipping frame")

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            **self.stats,
            'queue_size': self.frame_queue.qsize(),
            'total_classes': len(self.class_map),
        }

    def add_ontology_entry(self, prompt: str, class_name: str):
        """Add a new prompt->class mapping to the ontology."""
        self.ontology[prompt] = class_name
        if class_name not in self.class_map:
            self.class_map[class_name] = len(self.class_map)
            self._save_class_map()
            self._save_dataset_yaml()
        print(f"[Grounded] Added: '{prompt}' -> {class_name}")


# Quick test
if __name__ == "__main__":
    print("Grounded Labeler - Quick Test")
    print("=" * 50)

    labeler = GroundedLabeler(
        output_dir="data/grounded_test",
        use_florence=True,
        use_sam2_refinement=False  # Faster test
    )

    print(f"\nOntology ({len(labeler.ontology)} entries):")
    for prompt, cls in list(labeler.ontology.items())[:5]:
        print(f"  '{prompt}' -> {cls}")
    print("  ...")

    print(f"\nClass map ({len(labeler.class_map)} classes):")
    for cls, idx in labeler.class_map.items():
        print(f"  {idx}: {cls}")
