"""
Bootstrap Labeler: Build YOLO training data from scratch using FastSAM + LLM

Three-phase system:
1. DISCOVERY: FastSAM segments everything, LLM names each segment
2. VALIDATION: Run both SAM and YOLO, compare detections, add misses to training
3. YOLO-ONLY: YOLO is mature, use low-confidence verification only

Supports:
- Local LLM via LM Studio (fast, free, no rate limits) - RECOMMENDED
- Gemini API (rate limited, but more accurate)

Usage:
    from training.bootstrap_labeler import BootstrapLabeler

    # Phase 1 with Local LLM (recommended)
    labeler = BootstrapLabeler(phase="discovery", llm_mode="local")
    labeler.start()

    # Phase 1 with Gemini (slower, rate limited)
    labeler = BootstrapLabeler(phase="discovery", llm_mode="gemini")
"""

import os
import time
import json
import queue
import threading
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2


class BootstrapLabeler:
    """
    Bootstrap YOLO from scratch using FastSAM + LLM.

    Supports two LLM backends:
    1. Local LLM (LM Studio) - Fast, free, no rate limits
    2. Gemini API - Rate limited but more accurate

    Phase 1 (Discovery):
    - FastSAM segments all objects in frame
    - LLM names each segment
    - Save as YOLO training data

    Phase 2 (Validation):
    - Run FastSAM and YOLO in parallel
    - Compare: what did YOLO miss?
    - Add missed objects to training data
    - Auto-transition to Phase 3 when ready
    """

    # Phase thresholds
    MIN_SAMPLES_FOR_VALIDATION = 100
    YOLO_ACCURACY_THRESHOLD = 0.85
    VALIDATION_WINDOW = 50

    # Segment filtering
    MIN_SEGMENT_AREA = 800          # Increased to filter tiny HUD buttons (was 400)
    MAX_SEGMENT_AREA = 500000       # Large max to catch player ship + UI windows
    MIN_DIMENSION = 25              # Minimum width OR height in pixels - filters tiny icons
    MIN_ASPECT_RATIO = 0.15         # More lenient for odd shapes
    MAX_ASPECT_RATIO = 7.0          # More lenient for wide UI elements

    # Debug - save crops for review
    DEBUG_SAVE_CROPS = True

    def __init__(self,
                 output_dir: str = "data/bootstrap",
                 phase: str = "discovery",
                 llm_mode: str = "local",  # "local" or "gemini"
                 local_url: str = "http://localhost:1234",
                 local_model: str = "local-model",
                 gemini_api_key: str = None,
                 gemini_model: str = "gemini-3-pro-preview",
                 fastsam_model: str = "FastSAM-s.pt",
                 segmenter: str = "sam3",  # "sam3" (best), "sam2", or "fastsam"
                 max_queue_size: int = 10,
                 sample_interval: int = 30,
                 use_gallery: bool = True,
                 crop_padding: float = 0.5,
                 sam3_points_per_side: int = 16,
                 sam3_batch_size: int = 32):
        """
        Args:
            output_dir: Where to save training data
            phase: "discovery" or "validation"
            llm_mode: "local" (LM Studio) or "gemini" (Google API)
            local_url: LM Studio server URL
            local_model: Model name in LM Studio
            gemini_api_key: Google API key (if using gemini)
            gemini_model: Gemini model name
            fastsam_model: FastSAM model path
            segmenter: "sam3" (best, requires sam3.pt), "sam2", or "fastsam" (fastest)
            max_queue_size: Max frames in queue
            sample_interval: Process every N frames
            use_gallery: Whether to use gallery references (disable for clean initial collection)
            crop_padding: Padding around crops as fraction of bbox size (0.5 = 50%)
            sam3_points_per_side: Grid density for SAM3 point prompts (16 = 256 points)
            sam3_batch_size: Batch size for SAM3 point processing (lower = less VRAM)
        """
        self.output_dir = Path(output_dir)
        self.phase = phase
        self.llm_mode = llm_mode
        self.local_url = local_url.rstrip('/')
        self.local_model = local_model
        self.gemini_model = gemini_model
        self.fastsam_model = fastsam_model
        self.segmenter = segmenter
        self.sample_interval = sample_interval
        self.use_gallery = use_gallery
        self.crop_padding = crop_padding
        self.sam3_points_per_side = sam3_points_per_side
        self.sam3_batch_size = sam3_batch_size

        # API key for Gemini
        self.api_key = gemini_api_key or os.environ.get('GOOGLE_API') or os.environ.get('GOOGLE_API_KEY')

        # Debug: show API key status
        if self.llm_mode == "gemini":
            if self.api_key:
                print(f"[Bootstrap] ✅ Gemini API key found: {self.api_key[:10]}...")
            else:
                print("[Bootstrap] ⚠️ No Gemini API key found! Set GOOGLE_API env var or pass gemini_api_key")

        # Create directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.gallery_dir = self.output_dir / "gallery"
        self.debug_dir = self.output_dir / "debug_crops"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        if self.DEBUG_SAVE_CROPS:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Class mapping
        self.class_map = {}
        self._load_class_map()

        # Gallery
        self.gallery = {}
        self._load_gallery()

        # Threading
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread = None

        # Frame counter
        self.frame_counter = 0
        self.debug_crop_counter = 0

        # Stats
        self.stats = {
            'frames_processed': 0,
            'segments_found': 0,
            'objects_labeled': 0,
            'llm_calls': 0,
            'yolo_matches': 0,
            'yolo_misses': 0,
            'phase': phase,
            'llm_mode': llm_mode,
            'start_time': None
        }

        # Validation tracking
        self.validation_history = []

        # Models (lazy init) - separate variables for each model type
        self._fastsam = None      # FastSAM model (Ultralytics FastSAM)
        self._sam3_model = None   # SAM3 model (Ultralytics SAM)
        self._sam2_generator = None  # SAM2 AutomaticMaskGenerator (facebook sam2)
        self._use_sam2 = False    # Will be set to True if SAM2 loads successfully
        self._use_sam3 = False    # Will be set to True if SAM3 loads successfully
        self._gemini = None
        self._gemini_client = None  # New google.genai client

        # Rate limiting for Gemini only
        self._last_gemini_call = 0
        self.GEMINI_MIN_INTERVAL = 0.5

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

    def _load_gallery(self):
        """Load reference gallery (always loads for saving, use_gallery controls display)."""
        self.gallery = {}
        if not self.use_gallery:
            print("[Bootstrap] Gallery references disabled - model won't see examples")
        if not self.gallery_dir.exists():
            return
        for class_dir in self.gallery_dir.iterdir():
            if class_dir.is_dir():
                crops = list(class_dir.glob("*.jpg"))
                if crops:
                    self.gallery[class_dir.name] = [str(p) for p in crops[:5]]

    def _get_fastsam(self):
        """Lazy-load FastSAM model (Ultralytics version)."""
        if self._fastsam is None:
            try:
                from ultralytics import FastSAM

                model_paths = [
                    self.fastsam_model,
                    f"models/{self.fastsam_model}",
                    f"F:/dev/bot/{self.fastsam_model}",
                    f"F:/dev/bot/models/{self.fastsam_model}",
                ]

                model_path = None
                for p in model_paths:
                    if Path(p).exists():
                        model_path = p
                        break

                if model_path is None:
                    model_path = self.fastsam_model
                    print(f"[Bootstrap] FastSAM model will be downloaded: {model_path}")

                self._fastsam = FastSAM(model_path)
                self._use_sam2 = False
                print(f"[Bootstrap] FastSAM loaded: {model_path}")

            except ImportError:
                print("[Bootstrap] Ultralytics not installed. Run: pip install ultralytics")
                return None
            except Exception as e:
                print(f"[Bootstrap] FastSAM load error: {e}")
                return None

        return self._fastsam

    def _get_sam2(self):
        """Lazy-load SAM2 model with automatic mask generator."""
        if self._sam2_generator is None:
            try:
                import torch
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

                # SAM2 model paths - support both sam2 and sam2.1 naming conventions
                model_configs = [
                    ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
                    ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
                    ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
                    ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
                ]

                # Try to find an available checkpoint
                checkpoint_path = None
                config_name = None

                for cfg, ckpt in model_configs:
                    paths_to_try = [
                        f"F:/dev/bot/darkorbit_bot/models/{ckpt}",
                        f"models/{ckpt}",
                        ckpt,
                    ]
                    for p in paths_to_try:
                        if Path(p).exists():
                            checkpoint_path = p
                            config_name = cfg
                            break
                    if checkpoint_path:
                        break

                if checkpoint_path is None:
                    print(f"[Bootstrap] SAM2 checkpoint not found")
                    print(f"[Bootstrap] Looking for: sam2_hiera_tiny.pt or sam2.1_hiera_tiny.pt")
                    print(f"[Bootstrap] Download from: https://github.com/facebookresearch/sam2")
                    return None

                print(f"[Bootstrap] Loading SAM2: {checkpoint_path} (config: {config_name})")

                # Build SAM2 model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam2_model = build_sam2(config_name, checkpoint_path, device=device)

                # Create automatic mask generator - more masks for better coverage
                self._sam2_generator = SAM2AutomaticMaskGenerator(
                    model=sam2_model,
                    points_per_side=32,           # More points = more masks (32x32 = 1024 points)
                    points_per_batch=64,          # Process points in batches
                    pred_iou_thresh=0.7,          # IoU threshold for mask quality
                    stability_score_thresh=0.8,   # Stability threshold
                    box_nms_thresh=0.5,           # NMS threshold for overlapping boxes
                    crop_n_layers=0,              # No multi-crop for speed
                    min_mask_region_area=150,     # Lower to catch smaller objects
                )
                self._use_sam2 = True
                print(f"[Bootstrap] SAM2 AutomaticMaskGenerator loaded on {device}")

            except ImportError as e:
                print(f"[Bootstrap] SAM2 not installed: {e}")
                print("[Bootstrap] Run: pip install sam2")
                return None
            except Exception as e:
                print(f"[Bootstrap] SAM2 load error: {e}")
                import traceback
                traceback.print_exc()
                return None

        return self._sam2_generator

    def _get_sam3(self):
        """Lazy-load SAM3 model via Ultralytics."""
        if self._sam3_model is None:
            try:
                from ultralytics import SAM

                # SAM3 model paths
                sam3_paths = [
                    "sam3.pt",
                    "models/sam3.pt",
                    "F:/dev/bot/models/sam3.pt",
                    "F:/dev/bot/darkorbit_bot/models/sam3.pt",
                ]

                model_path = None
                for p in sam3_paths:
                    if Path(p).exists():
                        model_path = p
                        break

                if model_path is None:
                    print("[Bootstrap] SAM3 checkpoint not found (sam3.pt)")
                    print("[Bootstrap] Download from HuggingFace (requires access request):")
                    print("[Bootstrap] https://huggingface.co/facebook/sam3")
                    return None  # Return None, let caller handle fallback

                print(f"[Bootstrap] Loading SAM3: {model_path}")
                self._sam3_model = SAM(model_path)
                self._use_sam3 = True
                print("[Bootstrap] SAM3 loaded via Ultralytics")

            except ImportError as e:
                print(f"[Bootstrap] SAM3/Ultralytics not available: {e}")
                print("[Bootstrap] Install: pip install ultralytics>=8.3.237")
                return None
            except Exception as e:
                print(f"[Bootstrap] SAM3 load error: {e}")
                return None

        return self._sam3_model

    def _get_gemini(self):
        """Lazy-init Gemini using new google.genai SDK."""
        if self._gemini is None:
            if not self.api_key:
                print("[Bootstrap] ❌ No Gemini API key found. Set GOOGLE_API or GOOGLE_API_KEY env var.")
                return None
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
                self._gemini = self.gemini_model  # Store model name, client handles it
                print(f"[Bootstrap] Gemini initialized (new SDK): {self.gemini_model}")
            except ImportError:
                print("[Bootstrap] ❌ google-genai not installed. Run: pip install google-genai")
                return None
            except Exception as e:
                print(f"[Bootstrap] Gemini init error: {e}")
                return None
        return self._gemini

    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode image to base64."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _segment_frame(self, image: np.ndarray) -> List[Dict]:
        """Segment all objects in frame using SAM3, SAM2, or FastSAM."""
        if self.segmenter == "sam3":
            return self._segment_with_sam3(image)
        elif self.segmenter == "sam2":
            return self._segment_with_sam2(image)
        else:
            return self._segment_with_fastsam(image)

    def _segment_with_sam3(self, image: np.ndarray) -> List[Dict]:
        """Use SAM3 via Ultralytics to segment all objects.

        Since Ultralytics SAM doesn't have automatic mask generation like SAM2,
        we use a dense point grid to prompt segmentation across the image.
        Points are processed in batches to avoid CUDA OOM errors.
        """
        sam3_model = self._get_sam3()
        if sam3_model is None:
            return []

        try:
            import torch
            h, w = image.shape[:2]

            # Generate a dense grid of points to prompt SAM3
            # More points = more potential masks (but slower)
            points_per_side = getattr(self, 'sam3_points_per_side', 16)  # 16x16 = 256 points

            # Create point grid
            x_coords = np.linspace(0, w - 1, points_per_side).astype(int)
            y_coords = np.linspace(0, h - 1, points_per_side).astype(int)

            # Build point prompts - each point is [x, y]
            all_points = []
            for y in y_coords:
                for x in x_coords:
                    all_points.append([x, y])

            all_points = np.array(all_points)
            total_points = len(all_points)

            # Process in batches to avoid CUDA OOM (batch_size points at a time)
            batch_size = getattr(self, 'sam3_batch_size', 32)  # Process 32 points at a time
            print(f"[Bootstrap] Running SAM3 with {total_points} point prompts ({points_per_side}x{points_per_side} grid) in batches of {batch_size}...")

            all_results = []
            for batch_start in range(0, total_points, batch_size):
                batch_end = min(batch_start + batch_size, total_points)
                batch_points = all_points[batch_start:batch_end]
                batch_labels = np.ones(len(batch_points), dtype=int)

                # Clear CUDA cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Run SAM3 with batch of point prompts
                results = sam3_model(image, points=batch_points, labels=batch_labels, verbose=False)
                if results:
                    all_results.extend(results)

            results = all_results

            if not results or len(results) == 0:
                print("[Bootstrap] SAM3 found no masks")
                return []

            # Extract masks from results
            masks_data = []
            for result in results:
                if result.masks is None:
                    continue
                for i, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()

                    # IMPORTANT: Resize mask to original image size if different
                    mask_h, mask_w = mask_np.shape[:2]
                    if mask_h != h or mask_w != w:
                        mask_np = cv2.resize(mask_np.astype(np.float32), (w, h),
                                            interpolation=cv2.INTER_LINEAR) > 0.5

                    mask_np = mask_np.astype(bool)

                    # Get bounding box from mask
                    ys, xs = np.where(mask_np)
                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    x1, y1 = xs.min(), ys.min()
                    x2, y2 = xs.max(), ys.max()
                    bw, bh = x2 - x1, y2 - y1
                    area = mask_np.sum()

                    masks_data.append({
                        'segmentation': mask_np,
                        'bbox': [x1, y1, bw, bh],
                        'area': area,
                        'predicted_iou': 0.9,  # SAM3 masks are high quality
                        'stability_score': 0.9,
                    })

            print(f"[Bootstrap] SAM3 found {len(masks_data)} raw masks")

            # Apply NMS to remove duplicate overlapping masks (from point grid)
            masks_data = self._nms_masks(masks_data, iou_threshold=0.7)
            print(f"[Bootstrap] SAM3 after NMS: {len(masks_data)} unique masks")

            # Apply same filtering as SAM2
            segments = []
            filtered_area = 0
            filtered_dimension = 0
            filtered_aspect = 0

            # Use configurable padding
            crop_padding = getattr(self, 'crop_padding', 0.5)

            for mask_info in masks_data:
                mask_np = mask_info['segmentation']
                x1, y1, bw, bh = mask_info['bbox']
                area = mask_info['area']

                # Filter by area
                if area < self.MIN_SEGMENT_AREA or area > self.MAX_SEGMENT_AREA:
                    filtered_area += 1
                    continue

                # Filter tiny objects
                if bw < self.MIN_DIMENSION and bh < self.MIN_DIMENSION:
                    filtered_dimension += 1
                    continue

                aspect = bw / max(bh, 1)
                if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
                    filtered_aspect += 1
                    continue

                # Configurable padding to capture context (nicknames, HP bars, etc.)
                x2, y2 = x1 + bw, y1 + bh
                pad_x = int(bw * crop_padding)
                pad_y = int(bh * crop_padding)
                crop_x1 = max(0, int(x1) - pad_x)
                crop_y1 = max(0, int(y1) - pad_y)
                crop_x2 = min(w, int(x2) + pad_x)
                crop_y2 = min(h, int(y2) + pad_y)
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                center_x = (x1 + bw/2) / w
                center_y = (y1 + bh/2) / h

                segments.append({
                    'bbox': [int(x1), int(y1), int(bw), int(bh)],
                    'crop': crop,
                    'mask': mask_np,
                    'area': area,
                    'center': (center_x, center_y)
                })

            self.stats['segments_found'] += len(segments)
            print(f"[Bootstrap] SAM3 after filtering: {len(segments)} valid segments")
            print(f"[Bootstrap]   Filtered out: {filtered_area} (area), {filtered_dimension} (tiny), {filtered_aspect} (aspect)")
            return segments

        except Exception as e:
            print(f"[Bootstrap] SAM3 segment error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to SAM2
            print("[Bootstrap] Falling back to SAM2...")
            return self._segment_with_sam2(image)

    def _segment_with_sam2(self, image: np.ndarray) -> List[Dict]:
        """Use SAM2 AutomaticMaskGenerator to segment all objects."""
        mask_generator = self._get_sam2()
        if mask_generator is None:
            return []

        try:
            h, w = image.shape[:2]

            # SAM2 expects RGB image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Generate masks automatically
            masks_data = mask_generator.generate(image_rgb)

            if not masks_data:
                return []

            print(f"[Bootstrap] SAM2 found {len(masks_data)} masks")

            segments = []
            filtered_area = 0
            filtered_dimension = 0
            filtered_aspect = 0
            filtered_quality = 0

            for mask_info in masks_data:
                mask_np = mask_info['segmentation']
                bbox = mask_info['bbox']  # [x, y, w, h]
                area = mask_info['area']
                iou_score = mask_info.get('predicted_iou', 0.5)
                stability = mask_info.get('stability_score', 0.5)

                x1, y1, bw, bh = bbox
                x2, y2 = x1 + bw, y1 + bh

                # Filter by area
                if area < self.MIN_SEGMENT_AREA or area > self.MAX_SEGMENT_AREA:
                    filtered_area += 1
                    continue

                # Filter tiny objects (HUD buttons, icons) - both dimensions must be reasonable
                if bw < self.MIN_DIMENSION and bh < self.MIN_DIMENSION:
                    filtered_dimension += 1
                    continue

                aspect = bw / max(bh, 1)
                if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
                    filtered_aspect += 1
                    continue

                # Be more lenient - let LLM decide if it's valid
                if iou_score < 0.4 or stability < 0.5:
                    filtered_quality += 1
                    continue

                # Larger padding to capture context (nicknames, HP bars, selection rings)
                pad_x = int(bw * 0.5)  # 50% padding instead of 20%
                pad_y = int(bh * 0.5)
                crop_x1 = max(0, int(x1) - pad_x)
                crop_y1 = max(0, int(y1) - pad_y)
                crop_x2 = min(w, int(x2) + pad_x)
                crop_y2 = min(h, int(y2) + pad_y)
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                # Calculate center position (normalized 0-1) for context
                center_x = (x1 + bw/2) / w
                center_y = (y1 + bh/2) / h

                segments.append({
                    'bbox': [int(x1), int(y1), int(bw), int(bh)],
                    'crop': crop,
                    'mask': mask_np,
                    'area': area,
                    'center': (center_x, center_y)  # Position on screen
                })

            self.stats['segments_found'] += len(segments)
            print(f"[Bootstrap] SAM2 after filtering: {len(segments)} valid segments")
            print(f"[Bootstrap]   Filtered out: {filtered_area} (area), {filtered_dimension} (tiny), {filtered_aspect} (aspect), {filtered_quality} (quality)")
            return segments

        except Exception as e:
            print(f"[Bootstrap] SAM2 segment error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _segment_with_fastsam(self, image: np.ndarray) -> List[Dict]:
        """Use FastSAM (Ultralytics) to segment all objects."""
        model = self._get_fastsam()
        if model is None:
            return []

        try:
            h, w = image.shape[:2]

            results = model(
                image,
                device='cuda',
                retina_masks=True,
                imgsz=640,
                conf=0.25,
                iou=0.7
            )

            if not results or len(results) == 0:
                return []

            result = results[0]
            if result.masks is None:
                return []

            masks = result.masks.data
            if masks is None or len(masks) == 0:
                return []

            print(f"[Bootstrap] FastSAM found {len(masks)} masks")

            segments = []
            for i in range(len(masks)):
                mask = masks[i]
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np.astype(np.float32), (w, h)) > 0.5

                coords = np.where(mask_np > 0)
                if len(coords[0]) == 0:
                    continue

                y1, y2 = int(coords[0].min()), int(coords[0].max())
                x1, x2 = int(coords[1].min()), int(coords[1].max())

                area = (x2 - x1) * (y2 - y1)
                if area < self.MIN_SEGMENT_AREA or area > self.MAX_SEGMENT_AREA:
                    continue

                aspect = (x2 - x1) / max(y2 - y1, 1)
                if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
                    continue

                pad_x = int((x2 - x1) * 0.2)
                pad_y = int((y2 - y1) * 0.2)
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w, x2 + pad_x)
                crop_y2 = min(h, y2 + pad_y)
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                segments.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'crop': crop,
                    'mask': mask_np,
                    'area': area
                })

            self.stats['segments_found'] += len(segments)
            print(f"[Bootstrap] FastSAM after filtering: {len(segments)} valid segments")
            return segments

        except Exception as e:
            print(f"[Bootstrap] FastSAM segment error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _classify_with_local_llm(self, crop: np.ndarray) -> Tuple[str, float]:
        """Use local LLM (LM Studio) to classify a segment."""
        try:
            self.stats['llm_calls'] += 1

            # Build comparison image
            comparison_img = self._build_comparison_image(crop)
            img_base64 = self._encode_image_base64(comparison_img)

            # Build prompt
            known = list(self.class_map.keys())
            prompt = f"""DarkOrbit game - classify the RED-FRAMED object.

Known classes so far: {', '.join(known) if known else 'None'}

IMPORTANT: If the object doesn't match any known class, CREATE A NEW descriptive class name!
Examples of good new class names: Drone, MiningLaser, SpaceStation, Asteroid, RepairBot, etc.
Don't force-fit into wrong categories - it's better to create a new class than misclassify.

HOW TO IDENTIFY SHIPS:
- PlayerShip/AllyPlayer: has GREEN nickname text above, green HP bar, green selection ring
- EnemyPlayer: has RED nickname text above, red HP bar, red selection ring
- NPC aliens (Streuner, Sibelon, etc): NO nickname OR yellow/orange text, often no HP bar visible

COMMON GAME OBJECTS:
- Drone: small flying companion near player ship, often metallic
- Streuner: small gray/blue triangular NPC alien
- Lordakia: medium red/orange NPC alien
- Saimon: green glowing NPC alien
- Sibelon: LARGE gray industrial NPC ship, multiple segments (NO nickname!)
- Kristallin: crystalline/purple NPC alien
- BonusBox: spinning golden/purple cube
- CargoBox: cargo container
- Portal: circular gate
- Planet: large sphere

UI WINDOWS: Minimap, ShipWindow, ChatWindow, MenuBar, Hotbar, GroupWindow, UserWindow, LogWindow

IGNORE: buttons, icons, text-only, background, stars, effects

KEY: Nickname text above = player ship. No nickname = NPC/object. Unknown object = CREATE NEW CLASS NAME.

JSON ONLY: {{"class": "NAME", "confidence": 0.9, "reason": "brief"}}"""

            # Call local LLM (no timeout - wait for response)
            response = requests.post(
                f"{self.local_url}/v1/chat/completions",
                json={
                    "model": self.local_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 200,
                    "temperature": 0.3
                },
                timeout=None  # Wait indefinitely for LM Studio response
            )

            if response.status_code != 200:
                print(f"[Bootstrap] Local LLM error: {response.status_code}")
                return "unknown", 0.0

            result_text = response.json()['choices'][0]['message']['content'].strip()

            # Parse JSON
            if "```" in result_text:
                parts = result_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        result_text = part
                        break

            result = json.loads(result_text)
            class_name = result.get('class', 'unknown')
            confidence = float(result.get('confidence', 0.5))

            # Save debug crop with classification
            if self.DEBUG_SAVE_CROPS:
                self.debug_crop_counter += 1
                debug_name = f"{self.debug_crop_counter:04d}_{class_name}_{confidence:.0f}.jpg"
                debug_path = self.debug_dir / debug_name
                cv2.imwrite(str(debug_path), crop)

            if class_name == "IGNORE":
                print(f"[Bootstrap] LLM → IGNORE (reason: {result.get('reason', 'n/a')})")
                return "IGNORE", 0.0

            # Add new class
            if class_name not in self.class_map:
                self.class_map[class_name] = len(self.class_map)
                self._save_class_map()
                print(f"[Bootstrap] 🆕 New class: {class_name}")
            else:
                print(f"[Bootstrap] LLM → {class_name} ({confidence:.0%})")

            return class_name, confidence

        except requests.exceptions.ConnectionError:
            print("[Bootstrap] ❌ Local LLM not running. Start LM Studio server.")
            return "unknown", 0.0
        except json.JSONDecodeError as e:
            print(f"[Bootstrap] ❌ JSON parse error: {e} - raw: {result_text[:100] if 'result_text' in dir() else 'n/a'}")
            return "unknown", 0.0
        except Exception as e:
            print(f"[Bootstrap] ❌ Local LLM error: {e}")
            return "unknown", 0.0

    def _classify_with_gemini(self, crop: np.ndarray) -> Tuple[str, float]:
        """Use Gemini API to classify a segment (rate limited)."""
        gemini = self._get_gemini()
        if gemini is None:
            print("[Bootstrap] ❌ Gemini not initialized (missing API key?)")
            return "unknown", 0.0

        # Rate limiting
        now = time.time()
        if now - self._last_gemini_call < self.GEMINI_MIN_INTERVAL:
            time.sleep(self.GEMINI_MIN_INTERVAL - (now - self._last_gemini_call))

        try:
            from PIL import Image

            self._last_gemini_call = time.time()
            self.stats['llm_calls'] += 1

            comparison_img = self._build_comparison_image(crop)

            # Debug: save what Gemini sees
            if self.DEBUG_SAVE_CROPS:
                self.debug_crop_counter += 1
                debug_comparison_path = self.debug_dir / f"{self.debug_crop_counter:04d}_comparison.jpg"
                cv2.imwrite(str(debug_comparison_path), comparison_img)

            comparison_pil = Image.fromarray(cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB))

            known = list(self.class_map.keys())
            prompt = f"""OUTPUT FORMAT: Return ONLY a JSON object. Example: {{"class": "Streuner", "confidence": 0.9, "reason": "small alien NPC"}}

DarkOrbit space game - classify the RED-FRAMED object.

Known classes so far: {', '.join(known) if known else 'None yet'}

IMPORTANT: If the object doesn't match any known class, CREATE A NEW descriptive class name!
Examples: Drone, MiningLaser, SpaceStation, Asteroid, RepairBot, Shield, Missile, etc.
Don't force-fit into wrong categories - creating a new accurate class is better than misclassifying.

HOW TO IDENTIFY SHIPS (look for visual cues):
- PlayerShip/AllyPlayer: GREEN nickname text above, green HP bar, green selection ring
- EnemyPlayer: RED nickname text above, red HP bar, red selection ring
- NPC aliens: NO nickname text OR yellow/orange text, often no HP bar

COMMON GAME OBJECTS:
- Drone: small flying companion near player ship, metallic appearance
- Streuner: small gray/blue triangular NPC alien
- Lordakia: medium red/orange NPC alien
- Saimon: green glowing NPC alien
- Sibelon: LARGE industrial gray NPC ship, multiple segments (NO nickname!)
- Kristallin: crystalline/purple NPC alien
- BonusBox: spinning golden/purple cube IN SPACE
- CargoBox: cargo container in space
- Portal: circular gate/portal structure
- Planet: large spherical celestial body

UI WINDOWS (dark panels with borders):
- Minimap, ShipWindow, ChatWindow, MenuBar, Hotbar, GroupWindow, UserWindow, LogWindow

IGNORE (output class="IGNORE"):
- Small buttons/icons inside UI windows
- Text, numbers, visual effects
- Background/stars/space
- Unrecognizable or blurry objects

KEY: Unknown object that's clearly something = CREATE NEW CLASS. Don't guess wrong.

JSON ONLY: {{"class": "NAME", "confidence": 0.0-1.0, "reason": "brief"}}"""

            # New google.genai SDK API with JSON mode
            from google.genai import types
            response = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[comparison_pil, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,  # Low temp for consistent JSON
                )
            )

            # Check if response has text
            if not response or not hasattr(response, 'text') or not response.text:
                print(f"[Bootstrap] ⚠️ Gemini returned empty response")
                # Check for safety blocks or other issues
                if hasattr(response, 'candidates') and response.candidates:
                    for c in response.candidates:
                        if hasattr(c, 'finish_reason'):
                            print(f"[Bootstrap]   Finish reason: {c.finish_reason}")
                        if hasattr(c, 'safety_ratings'):
                            print(f"[Bootstrap]   Safety: {c.safety_ratings}")
                return "unknown", 0.0

            result_text = response.text.strip()
            print(f"[Bootstrap] Raw response: {result_text[:200]}...")

            if "```" in result_text:
                parts = result_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        result_text = part
                        break

            # Handle empty result after parsing
            if not result_text or not result_text.startswith("{"):
                print(f"[Bootstrap] ⚠️ Could not extract JSON from response")
                return "unknown", 0.0

            result = json.loads(result_text)
            class_name = result.get('class', 'unknown')
            confidence = float(result.get('confidence', 0.5))

            # Save debug crop with classification
            if self.DEBUG_SAVE_CROPS:
                self.debug_crop_counter += 1
                debug_name = f"{self.debug_crop_counter:04d}_{class_name}_{confidence:.0f}.jpg"
                debug_path = self.debug_dir / debug_name
                cv2.imwrite(str(debug_path), crop)

            if class_name == "IGNORE":
                print(f"[Bootstrap] Gemini → IGNORE (reason: {result.get('reason', 'n/a')})")
                return "IGNORE", 0.0

            if class_name not in self.class_map:
                self.class_map[class_name] = len(self.class_map)
                self._save_class_map()
                print(f"[Bootstrap] 🆕 New class: {class_name}")
            else:
                print(f"[Bootstrap] Gemini → {class_name} ({confidence:.0%})")

            return class_name, confidence

        except Exception as e:
            print(f"[Bootstrap] ❌ Gemini error: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0

    def _classify_segment(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a segment using configured LLM."""
        if self.llm_mode == "local":
            return self._classify_with_local_llm(crop)
        else:
            return self._classify_with_gemini(crop)

    def _build_comparison_image(self, crop: np.ndarray) -> np.ndarray:
        """Build comparison image with target and optionally gallery."""
        # Preserve aspect ratio, fit into max 200x200
        max_size = 200
        h, w = crop.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        target = cv2.resize(crop, (new_w, new_h))

        # Add red border
        cv2.rectangle(target, (0, 0), (new_w-1, new_h-1), (0, 0, 255), 3)
        cv2.putText(target, "CLASSIFY THIS", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        target_size = max(new_w, new_h)

        # Skip gallery panel if use_gallery=False or no gallery yet
        if not self.use_gallery or not self.gallery:
            return target

        ref_size = 64  # Larger reference images for better comparison
        classes = list(self.gallery.items())[:6]  # Show up to 6 classes
        panel_h = len(classes) * (ref_size + 25)
        panel_w = 3 * ref_size + 20

        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        for row, (name, paths) in enumerate(classes):
            y = row * (ref_size + 25)
            cv2.putText(panel, name[:15], (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            for col, path in enumerate(paths[:3]):
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (ref_size, ref_size))
                        x = col * (ref_size + 5) + 5
                        panel[y+20:y+20+ref_size, x:x+ref_size] = img
                except:
                    pass

        final_h = max(new_h, panel_h)
        final_w = new_w + panel_w + 20
        combined = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        combined[:] = (20, 20, 20)
        combined[:new_h, :new_w] = target
        combined[:panel_h, new_w+20:new_w+20+panel_w] = panel

        return combined

    def _add_to_gallery(self, class_name: str, crop: np.ndarray):
        """Add crop to reference gallery (always saves, regardless of use_gallery)."""
        # Always save to gallery - use_gallery only controls whether we SHOW it to model
        if class_name not in self.gallery:
            self.gallery[class_name] = []

        if len(self.gallery[class_name]) >= 5:
            return

        class_dir = self.gallery_dir / class_name
        class_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = class_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.gallery[class_name].append(str(path))

    # Padding to add around bounding boxes (10% on each side) - used as fallback
    BBOX_PADDING = 0.15

    # Use polygon segmentation instead of bounding boxes
    USE_POLYGON_MASKS = True

    # Simplify polygon to reduce points (epsilon for cv2.approxPolyDP)
    POLYGON_SIMPLIFY_EPSILON = 2.0

    def _mask_to_polygon(self, mask: np.ndarray, simplify: bool = True) -> List[List[float]]:
        """
        Convert binary mask to polygon coordinates.

        Args:
            mask: Binary mask (H, W) boolean or uint8
            simplify: Whether to simplify polygon to reduce points

        Returns:
            List of polygons, each polygon is [x1, y1, x2, y2, ...] normalized
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

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify polygon if requested
        if simplify:
            epsilon = self.POLYGON_SIMPLIFY_EPSILON
            contour = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 3 points for a polygon
        if len(contour) < 3:
            return []

        # Convert to normalized coordinates [x1, y1, x2, y2, ...]
        polygon = []
        for point in contour:
            px, py = point[0]
            # Normalize to 0-1
            nx = px / w
            ny = py / h
            # Clamp to valid range
            nx = max(0, min(1, nx))
            ny = max(0, min(1, ny))
            polygon.extend([nx, ny])

        return polygon

    def _save_training_sample(self, image: np.ndarray, detections: List[Dict], reason: str):
        """Save image and YOLO-format labels (polygon segmentation or bbox)."""
        if not detections:
            print(f"[Bootstrap] ⚠️ _save_training_sample called with empty detections")
            return

        h, w = image.shape[:2]
        print(f"[Bootstrap] 💾 Saving training sample: {len(detections)} detections, image {w}x{h}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{reason}_{timestamp}"

        img_path = self.images_dir / f"{base_name}.jpg"
        success = cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[Bootstrap]   Image saved: {img_path} (success={success})")

        label_path = self.labels_dir / f"{base_name}.txt"
        labels_written = 0
        polygons_used = 0
        boxes_used = 0

        with open(label_path, 'w') as f:
            for det in detections:
                class_name = det['class_name']
                if class_name not in self.class_map:
                    print(f"[Bootstrap]   ⚠️ Class '{class_name}' not in class_map, skipping")
                    continue

                class_id = self.class_map[class_name]

                # Try to use polygon mask if available
                if self.USE_POLYGON_MASKS and 'mask' in det and det['mask'] is not None:
                    polygon = self._mask_to_polygon(det['mask'])
                    if polygon and len(polygon) >= 6:  # At least 3 points (6 coords)
                        # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                        coords_str = ' '.join(f"{c:.6f}" for c in polygon)
                        f.write(f"{class_id} {coords_str}\n")
                        labels_written += 1
                        polygons_used += 1
                        self.stats['objects_labeled'] += 1
                        continue

                # Fallback to bounding box
                bbox = det['bbox']
                x1, y1, bw, bh = bbox

                # Add padding to bounding box
                pad_w = bw * self.BBOX_PADDING
                pad_h = bh * self.BBOX_PADDING
                x1_padded = max(0, x1 - pad_w)
                y1_padded = max(0, y1 - pad_h)
                bw_padded = min(w - x1_padded, bw + 2 * pad_w)
                bh_padded = min(h - y1_padded, bh + 2 * pad_h)

                cx = (x1_padded + bw_padded/2) / w
                cy = (y1_padded + bh_padded/2) / h
                nw = bw_padded / w
                nh = bh_padded / h

                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                if nw > 0.01 and nh > 0.01:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    labels_written += 1
                    boxes_used += 1
                    self.stats['objects_labeled'] += 1
                else:
                    print(f"[Bootstrap]   ⚠️ Box too small: {class_name} nw={nw:.4f} nh={nh:.4f}")

        print(f"[Bootstrap]   Labels written: {labels_written} ({polygons_used} polygons, {boxes_used} boxes) to {label_path}")

    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter

        return inter / max(union, 1e-6)

    def _nms_masks(self, masks_data: List[Dict], iou_threshold: float = 0.7) -> List[Dict]:
        """Non-Maximum Suppression for masks to remove duplicates.

        When using point grid prompts, multiple points may generate the same mask.
        This removes duplicates by keeping larger/higher-quality masks.
        """
        if len(masks_data) <= 1:
            return masks_data

        # Sort by area (larger first - tend to be better quality)
        sorted_masks = sorted(masks_data, key=lambda x: x['area'], reverse=True)

        keep = []
        suppressed = [False] * len(sorted_masks)

        for i, mask_i in enumerate(sorted_masks):
            if suppressed[i]:
                continue

            keep.append(mask_i)
            box_i = mask_i['bbox']

            # Suppress all masks that overlap too much with this one
            for j in range(i + 1, len(sorted_masks)):
                if suppressed[j]:
                    continue

                box_j = sorted_masks[j]['bbox']
                iou = self._iou(box_i, box_j)

                if iou >= iou_threshold:
                    suppressed[j] = True

        return keep

    def _merge_overlapping_detections(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping detections of the same class.

        Fixes over-segmentation where one object gets split into multiple segments.
        If two detections of the same class overlap significantly, merge their bboxes.
        """
        if not detections:
            return detections

        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class_name']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)

        merged = []
        for cls, class_dets in by_class.items():
            if len(class_dets) == 1:
                merged.extend(class_dets)
                continue

            # Merge overlapping boxes within this class
            used = [False] * len(class_dets)
            for i, det1 in enumerate(class_dets):
                if used[i]:
                    continue

                # Find all overlapping detections
                group = [det1]
                used[i] = True

                for j, det2 in enumerate(class_dets):
                    if used[j]:
                        continue
                    if self._iou(det1['bbox'], det2['bbox']) >= iou_threshold:
                        group.append(det2)
                        used[j] = True

                if len(group) == 1:
                    merged.append(group[0])
                else:
                    # Merge bboxes: take union of all boxes
                    x1 = min(d['bbox'][0] for d in group)
                    y1 = min(d['bbox'][1] for d in group)
                    x2 = max(d['bbox'][0] + d['bbox'][2] for d in group)
                    y2 = max(d['bbox'][1] + d['bbox'][3] for d in group)

                    # Use highest confidence
                    best_conf = max(d['confidence'] for d in group)

                    merged.append({
                        'class_name': cls,
                        'confidence': best_conf,
                        'bbox': [x1, y1, x2 - x1, y2 - y1]
                    })

        return merged

    def process_frame_discovery(self, image: np.ndarray):
        """Phase 1: Segment and classify everything."""
        segments = self._segment_frame(image)

        if not segments:
            print("[Bootstrap] No segments found in frame")
            return

        print(f"[Bootstrap] Processing frame: {len(segments)} segments to classify...")
        detections = []
        for i, seg in enumerate(segments):
            print(f"[Bootstrap]   Segment {i+1}/{len(segments)} (area: {seg['area']}, bbox: {seg['bbox']})")
            class_name, conf = self._classify_segment(seg['crop'])

            if class_name == "IGNORE" or class_name == "unknown":
                print(f"[Bootstrap]     → Skipped ({class_name})")
                continue

            print(f"[Bootstrap]     → Added: {class_name} ({conf:.0%})")
            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'bbox': seg['bbox']
            })

            if conf >= 0.8:
                self._add_to_gallery(class_name, seg['crop'])

        # Merge overlapping detections of the same class (fixes over-segmentation)
        merged_detections = self._merge_overlapping_detections(detections)
        if len(merged_detections) < len(detections):
            print(f"[Bootstrap] Merged {len(detections)} → {len(merged_detections)} detections (combined overlapping)")

        print(f"[Bootstrap] Frame complete: {len(merged_detections)} valid detections out of {len(segments)} segments")
        if merged_detections:
            self._save_training_sample(image, merged_detections, "discovery_final")
            print(f"[Bootstrap] ✅ Saved frame with {len(merged_detections)} labeled objects")
        else:
            print(f"[Bootstrap] ⚠️ No valid detections to save")

    def process_frame_validation(self, image: np.ndarray, yolo_detections: List[Dict]):
        """Phase 2: Compare SAM vs YOLO, add misses to training."""
        segments = self._segment_frame(image)

        if not segments:
            return

        sam_detections = []
        for seg in segments:
            class_name, conf = self._classify_segment(seg['crop'])
            if class_name != "IGNORE" and class_name != "unknown":
                sam_detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': seg['bbox'],
                    'crop': seg['crop']
                })

        matched = 0
        missed = []

        for sam_det in sam_detections:
            found = False
            for yolo_det in yolo_detections:
                iou = self._iou(sam_det['bbox'], yolo_det.get('bbox', [0,0,0,0]))
                if iou > 0.5:
                    found = True
                    break

            if found:
                matched += 1
                self.stats['yolo_matches'] += 1
            else:
                missed.append(sam_det)
                self.stats['yolo_misses'] += 1

        if missed:
            self._save_training_sample(image, missed, "yolo_miss")
            for det in missed:
                if det['confidence'] >= 0.8:
                    self._add_to_gallery(det['class_name'], det['crop'])

        if sam_detections:
            self.validation_history.append({
                'sam_count': len(sam_detections),
                'matched': matched,
                'missed': len(missed)
            })

            if len(self.validation_history) >= self.VALIDATION_WINDOW:
                recent = self.validation_history[-self.VALIDATION_WINDOW:]
                total_sam = sum(r['sam_count'] for r in recent)
                total_matched = sum(r['matched'] for r in recent)

                if total_sam > 0:
                    accuracy = total_matched / total_sam
                    if accuracy >= self.YOLO_ACCURACY_THRESHOLD:
                        print(f"[Bootstrap] 🎉 YOLO accuracy {accuracy:.0%} >= {self.YOLO_ACCURACY_THRESHOLD:.0%}")
                        print("[Bootstrap] Ready for Phase 3 (YOLO-only mode)")
                        self.stats['phase'] = 'ready_for_yolo'

    def check_and_queue(self, image: np.ndarray, yolo_detections: List[Dict] = None) -> bool:
        """Queue frame for processing if interval reached."""
        self.frame_counter += 1

        if self.frame_counter % self.sample_interval != 0:
            return False

        return self.queue_frame(image, yolo_detections)

    def queue_frame(self, image: np.ndarray, yolo_detections: List[Dict] = None) -> bool:
        """Queue frame for processing immediately (bypasses interval check)."""
        if self.frame_queue.full():
            return False

        try:
            self.frame_queue.put_nowait({
                'image': image.copy(),
                'yolo_detections': yolo_detections or [],
                'timestamp': time.time()
            })
            return True
        except queue.Full:
            return False

    def _worker_loop(self):
        """Background worker."""
        while self.running:
            try:
                data = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if self.phase == "discovery":
                    self.process_frame_discovery(data['image'])
                else:
                    self.process_frame_validation(data['image'], data['yolo_detections'])

                self.stats['frames_processed'] += 1

            except Exception as e:
                print(f"[Bootstrap] Process error: {e}")
                import traceback
                traceback.print_exc()

    def start(self):
        """Start the labeler."""
        if self.running:
            return

        self.running = True
        self.stats['start_time'] = time.time()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        llm_info = f"Local LLM ({self.local_url})" if self.llm_mode == "local" else f"Gemini ({self.gemini_model})"
        print(f"[Bootstrap] Started in {self.phase.upper()} phase")
        print(f"   LLM: {llm_info}")
        print(f"   Sample interval: every {self.sample_interval} frames")
        print(f"   Known classes: {len(self.class_map)}")
        print(f"   Gallery images: {sum(len(v) for v in self.gallery.values())}")

    def stop(self):
        """Stop and save."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)

        self._save_class_map()
        self._save_dataset_yaml()
        self._print_stats()

    def _save_dataset_yaml(self):
        """Save dataset.yaml for YOLO training."""
        if not self.class_map:
            return

        yaml_path = self.output_dir / "dataset.yaml"
        names = {v: k for k, v in self.class_map.items()}
        max_id = max(names.keys()) if names else -1
        names_list = [names.get(i, f"class_{i}") for i in range(max_id + 1)]

        content = f"""# Bootstrap Dataset for YOLO Training
# Generated: {datetime.now().isoformat()}
# Phase: {self.phase}
# LLM: {self.llm_mode}
# Classes: {len(self.class_map)}

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
        print(f"  Bootstrap Labeler Statistics ({self.phase.upper()})")
        print("="*60)
        print(f"  Duration: {duration:.1f}s")
        print(f"  LLM mode: {self.llm_mode}")
        print(f"  Frames processed: {self.stats['frames_processed']}")
        print(f"  Segments found: {self.stats['segments_found']}")
        print(f"  Objects labeled: {self.stats['objects_labeled']}")
        print(f"  LLM calls: {self.stats['llm_calls']}")
        print(f"  Classes discovered: {len(self.class_map)}")

        if self.phase == "validation":
            total = self.stats['yolo_matches'] + self.stats['yolo_misses']
            if total > 0:
                acc = self.stats['yolo_matches'] / total
                print(f"  YOLO matches: {self.stats['yolo_matches']}")
                print(f"  YOLO misses: {self.stats['yolo_misses']}")
                print(f"  YOLO accuracy: {acc:.1%}")

        print(f"  Output: {self.output_dir}")
        print("="*60)

    def get_stats(self) -> Dict:
        """Get current stats."""
        return {
            **self.stats,
            'queue_size': self.frame_queue.qsize(),
            'total_classes': len(self.class_map),
            'gallery_size': sum(len(v) for v in self.gallery.values())
        }

    def should_transition_to_yolo(self) -> bool:
        """Check if YOLO is ready for Phase 3."""
        return self.stats.get('phase') == 'ready_for_yolo'
