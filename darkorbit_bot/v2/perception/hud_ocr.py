"""
DarkOrbit HUD OCR Reader

Extracts HP, Shield, and other values from the game HUD using OCR.
Designed to work with the standard DarkOrbit client layout.

HUD Layout (typical 1920x1080):
- Bottom-right corner: Main stats panel (HP, Shield, Credits, etc.)
- Top-right: Notifications
- Bottom: Hotbar and abilities
- Ships have HP/Shield bars above them (handled separately)

This module provides:
1. HUDReader - Main class for reading HUD values
2. Color-based bar detection (for HP/Shield bars)
3. OCR-based number extraction (for exact values)
"""

import numpy as np
import cv2
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import OCR libraries
EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass


@dataclass
class HUDConfig:
    """Configuration for HUD reading."""
    # Screen resolution (for region calculation)
    screen_width: int = 1920
    screen_height: int = 1080

    # HUD regions as fractions of screen (x1, y1, x2, y2)
    # These are normalized coordinates (0-1)
    stats_panel_region: Tuple[float, float, float, float] = (0.85, 0.85, 1.0, 1.0)  # Bottom-right
    hp_bar_region: Tuple[float, float, float, float] = (0.88, 0.88, 0.99, 0.91)  # HP bar approx
    shield_bar_region: Tuple[float, float, float, float] = (0.88, 0.91, 0.99, 0.94)  # Shield bar approx

    # Color ranges for bar detection (HSV)
    hp_color_low: Tuple[int, int, int] = (35, 100, 100)   # Green
    hp_color_high: Tuple[int, int, int] = (85, 255, 255)
    hp_color_critical_low: Tuple[int, int, int] = (0, 100, 100)  # Red when low
    hp_color_critical_high: Tuple[int, int, int] = (10, 255, 255)

    shield_color_low: Tuple[int, int, int] = (85, 100, 100)   # Cyan/Blue
    shield_color_high: Tuple[int, int, int] = (130, 255, 255)

    # OCR settings
    ocr_backend: str = "easyocr"  # "easyocr", "tesseract", "paddleocr", "color_only"
    ocr_confidence_threshold: float = 0.5

    # Caching
    cache_duration_ms: int = 100  # Don't re-OCR within this window

    # Smoothing
    smoothing_alpha: float = 0.3  # EMA smoothing for values

    # Debug
    debug_save_regions: bool = False
    debug_output_dir: str = "debug/hud_ocr"


@dataclass
class HUDValues:
    """Current HUD values."""
    # Player stats (0.0 - 1.0 normalized)
    hp: float = 1.0
    hp_max: int = 0  # Absolute max HP if detected
    hp_current: int = 0  # Absolute current HP if detected

    shield: float = 1.0
    shield_max: int = 0
    shield_current: int = 0

    # Other stats
    credits: int = 0
    uridium: int = 0
    experience: int = 0
    honor: int = 0

    # Confidence scores
    hp_confidence: float = 0.0
    shield_confidence: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Detection method used
    method: str = "none"  # "ocr", "color", "none"


class HUDReader:
    """
    Reads HUD values from DarkOrbit game screen.

    Supports multiple detection methods:
    1. Color-based bar detection (fast, works without OCR)
    2. OCR-based number extraction (accurate, needs OCR library)
    3. Hybrid approach (use color for bars, OCR for numbers)
    """

    def __init__(self, config: Optional[HUDConfig] = None):
        self.config = config or HUDConfig()

        # Initialize OCR engine
        self.ocr_engine = None
        self._init_ocr()

        # Value smoothing
        self._smoothed_hp = 1.0
        self._smoothed_shield = 1.0

        # Caching
        self._last_values: Optional[HUDValues] = None
        self._last_read_time = 0

        # Debug
        if self.config.debug_save_regions:
            Path(self.config.debug_output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"HUDReader initialized with backend: {self.config.ocr_backend}")

    def _init_ocr(self):
        """Initialize the OCR engine based on config."""
        backend = self.config.ocr_backend.lower()

        if backend == "easyocr" and EASYOCR_AVAILABLE:
            try:
                # GPU=False for faster startup, can change to True
                self.ocr_engine = easyocr.Reader(['en'], gpu=True, verbose=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to init EasyOCR: {e}, falling back to color-only")
                self.config.ocr_backend = "color_only"

        elif backend == "tesseract" and TESSERACT_AVAILABLE:
            # Tesseract doesn't need initialization
            self.ocr_engine = "tesseract"
            logger.info("Tesseract OCR ready")

        elif backend == "paddleocr" and PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to init PaddleOCR: {e}, falling back to color-only")
                self.config.ocr_backend = "color_only"
        else:
            if backend != "color_only":
                logger.warning(f"OCR backend '{backend}' not available, using color-only detection")
            self.config.ocr_backend = "color_only"

    def read(self, frame: np.ndarray, force: bool = False) -> HUDValues:
        """
        Read HUD values from a game frame.

        Args:
            frame: BGR image (screenshot)
            force: Force re-read even if cached

        Returns:
            HUDValues with current HP, shield, etc.
        """
        current_time = time.time() * 1000  # ms

        # Return cached if recent
        if not force and self._last_values is not None:
            if current_time - self._last_read_time < self.config.cache_duration_ms:
                return self._last_values

        values = HUDValues()

        # Method 1: Color-based bar detection (fast)
        hp_color, shield_color = self._detect_bars_by_color(frame)

        if hp_color is not None:
            values.hp = hp_color
            values.hp_confidence = 0.7  # Color detection is moderately confident
            values.method = "color"

        if shield_color is not None:
            values.shield = shield_color
            values.shield_confidence = 0.7

        # Method 2: OCR for exact numbers (if available)
        if self.config.ocr_backend != "color_only" and self.ocr_engine is not None:
            ocr_values = self._read_ocr(frame)

            # Use OCR values if confident
            if ocr_values.get('hp_confidence', 0) > self.config.ocr_confidence_threshold:
                values.hp = ocr_values.get('hp', values.hp)
                values.hp_current = ocr_values.get('hp_current', 0)
                values.hp_max = ocr_values.get('hp_max', 0)
                values.hp_confidence = ocr_values.get('hp_confidence', values.hp_confidence)
                values.method = "ocr"

            if ocr_values.get('shield_confidence', 0) > self.config.ocr_confidence_threshold:
                values.shield = ocr_values.get('shield', values.shield)
                values.shield_current = ocr_values.get('shield_current', 0)
                values.shield_max = ocr_values.get('shield_max', 0)
                values.shield_confidence = ocr_values.get('shield_confidence', values.shield_confidence)

        # Apply smoothing
        alpha = self.config.smoothing_alpha
        self._smoothed_hp = alpha * values.hp + (1 - alpha) * self._smoothed_hp
        self._smoothed_shield = alpha * values.shield + (1 - alpha) * self._smoothed_shield

        values.hp = self._smoothed_hp
        values.shield = self._smoothed_shield
        values.timestamp = time.time()

        # Cache
        self._last_values = values
        self._last_read_time = current_time

        return values

    def _get_region_pixels(self, frame: np.ndarray,
                           region: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract a region from the frame."""
        h, w = frame.shape[:2]
        x1 = int(region[0] * w)
        y1 = int(region[1] * h)
        x2 = int(region[2] * w)
        y2 = int(region[3] * h)

        # Clamp to frame bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        return frame[y1:y2, x1:x2].copy()

    def _detect_bars_by_color(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect HP and Shield bars by their color fill percentage.

        Returns:
            (hp_ratio, shield_ratio) - both 0.0 to 1.0, or None if not detected
        """
        hp_ratio = None
        shield_ratio = None

        # Extract HP bar region
        hp_region = self._get_region_pixels(frame, self.config.hp_bar_region)
        if hp_region.size > 0:
            hp_ratio = self._measure_bar_fill(
                hp_region,
                self.config.hp_color_low,
                self.config.hp_color_high,
                self.config.hp_color_critical_low,
                self.config.hp_color_critical_high
            )

        # Extract Shield bar region
        shield_region = self._get_region_pixels(frame, self.config.shield_bar_region)
        if shield_region.size > 0:
            shield_ratio = self._measure_bar_fill(
                shield_region,
                self.config.shield_color_low,
                self.config.shield_color_high
            )

        # Debug: save regions
        if self.config.debug_save_regions:
            ts = int(time.time() * 1000)
            if hp_region.size > 0:
                cv2.imwrite(f"{self.config.debug_output_dir}/hp_region_{ts}.png", hp_region)
            if shield_region.size > 0:
                cv2.imwrite(f"{self.config.debug_output_dir}/shield_region_{ts}.png", shield_region)

        return hp_ratio, shield_ratio

    def _measure_bar_fill(self, region: np.ndarray,
                          color_low: Tuple[int, int, int],
                          color_high: Tuple[int, int, int],
                          alt_color_low: Optional[Tuple[int, int, int]] = None,
                          alt_color_high: Optional[Tuple[int, int, int]] = None) -> Optional[float]:
        """
        Measure how much of a bar region is filled with the target color.

        Args:
            region: BGR image of the bar
            color_low/high: HSV range for primary color
            alt_color_low/high: HSV range for alternate color (e.g., red when critical)

        Returns:
            Fill ratio 0.0 to 1.0, or None if detection failed
        """
        if region.size == 0:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Create mask for primary color
        mask = cv2.inRange(hsv, np.array(color_low), np.array(color_high))

        # Add alternate color if provided
        if alt_color_low is not None and alt_color_high is not None:
            mask_alt = cv2.inRange(hsv, np.array(alt_color_low), np.array(alt_color_high))
            mask = cv2.bitwise_or(mask, mask_alt)

        # Calculate horizontal fill (bars usually fill left to right)
        h, w = mask.shape
        if w == 0:
            return None

        # For each column, check if there's color
        col_sums = np.sum(mask > 0, axis=0)
        threshold = h * 0.3  # At least 30% of column height has color

        filled_cols = np.sum(col_sums > threshold)
        fill_ratio = filled_cols / w

        # Alternative: measure total colored pixels
        total_pixels = mask.size
        colored_pixels = np.sum(mask > 0)
        pixel_ratio = colored_pixels / total_pixels if total_pixels > 0 else 0

        # Use the more reliable of the two
        # Horizontal scan is better for bars, pixel ratio as fallback
        if fill_ratio > 0.05:  # Detected something
            return fill_ratio
        elif pixel_ratio > 0.1:
            return pixel_ratio

        return None

    def _read_ocr(self, frame: np.ndarray) -> Dict:
        """
        Read numeric values using OCR.

        Returns:
            Dict with hp, shield, credits, etc. and confidence scores
        """
        result = {}

        # Extract stats panel region
        stats_region = self._get_region_pixels(frame, self.config.stats_panel_region)

        if stats_region.size == 0:
            return result

        # Preprocess for OCR
        processed = self._preprocess_for_ocr(stats_region)

        # Run OCR
        text_results = self._run_ocr(processed)

        # Parse results
        result = self._parse_ocr_results(text_results)

        return result

    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Preprocess image region for better OCR accuracy."""
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Upscale for better OCR (2x)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (white text on dark background)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return binary

    def _run_ocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Run OCR on preprocessed image.

        Returns:
            List of (text, confidence) tuples
        """
        results = []

        if self.config.ocr_backend == "easyocr" and isinstance(self.ocr_engine, easyocr.Reader):
            try:
                ocr_result = self.ocr_engine.readtext(image, detail=1)
                for detection in ocr_result:
                    text = detection[1]
                    confidence = detection[2] if len(detection) > 2 else 0.5
                    results.append((text, confidence))
            except Exception as e:
                logger.debug(f"EasyOCR error: {e}")

        elif self.config.ocr_backend == "tesseract":
            try:
                # Run Tesseract with digits-only config
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,./%()'
                text = pytesseract.image_to_string(image, config=custom_config)
                # Tesseract doesn't give per-word confidence easily
                for line in text.strip().split('\n'):
                    if line.strip():
                        results.append((line.strip(), 0.6))
            except Exception as e:
                logger.debug(f"Tesseract error: {e}")

        elif self.config.ocr_backend == "paddleocr" and self.ocr_engine is not None:
            try:
                ocr_result = self.ocr_engine.ocr(image, cls=False)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        results.append((text, confidence))
            except Exception as e:
                logger.debug(f"PaddleOCR error: {e}")

        return results

    def _parse_ocr_results(self, text_results: List[Tuple[str, float]]) -> Dict:
        """
        Parse OCR text to extract HP, Shield, etc.

        Common DarkOrbit formats:
        - HP: "123,456 / 200,000" or "123456/200000" or just "123456"
        - Shield: "50,000 / 100,000"
        - Credits: "1,234,567"
        """
        result = {
            'hp_confidence': 0.0,
            'shield_confidence': 0.0
        }

        full_text = ' '.join([t[0] for t in text_results])
        avg_confidence = np.mean([t[1] for t in text_results]) if text_results else 0.0

        # Pattern for "current / max" format
        ratio_pattern = r'(\d[\d,\.]*)\s*/\s*(\d[\d,\.]*)'

        # Pattern for single numbers
        number_pattern = r'(\d[\d,\.]+)'

        # Try to find HP/Shield patterns
        # Look for keywords
        hp_keywords = ['hp', 'health', 'life', 'hit']
        shield_keywords = ['shield', 'sh', 'armor']

        lines = full_text.lower().split('\n')

        for line in lines:
            # Check for ratio format
            ratio_match = re.search(ratio_pattern, line)
            if ratio_match:
                current = self._parse_number(ratio_match.group(1))
                maximum = self._parse_number(ratio_match.group(2))

                if maximum > 0:
                    ratio = current / maximum

                    # Determine if HP or Shield based on context or position
                    is_hp = any(kw in line for kw in hp_keywords)
                    is_shield = any(kw in line for kw in shield_keywords)

                    if is_hp or (not is_shield and result.get('hp') is None):
                        result['hp'] = ratio
                        result['hp_current'] = current
                        result['hp_max'] = maximum
                        result['hp_confidence'] = avg_confidence
                    elif is_shield:
                        result['shield'] = ratio
                        result['shield_current'] = current
                        result['shield_max'] = maximum
                        result['shield_confidence'] = avg_confidence

        return result

    def _parse_number(self, text: str) -> int:
        """Parse a number string that may contain commas/periods."""
        # Remove common separators
        cleaned = text.replace(',', '').replace('.', '').replace(' ', '')
        try:
            return int(cleaned)
        except ValueError:
            return 0

    def calibrate(self, frame: np.ndarray) -> Dict:
        """
        Auto-calibrate HUD regions by detecting bars and text.

        Call this once with a good screenshot to find optimal regions.

        Returns:
            Dict with detected regions and recommended config
        """
        h, w = frame.shape[:2]

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find green (HP) regions
        green_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find cyan (Shield) regions
        cyan_mask = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([130, 255, 255]))
        cyan_contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        calibration = {
            'screen_size': (w, h),
            'green_regions': [],
            'cyan_regions': [],
            'recommended_hp_region': None,
            'recommended_shield_region': None
        }

        # Analyze green contours (potential HP bars)
        for cnt in green_contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area > 100 and cw > ch * 2:  # Bar-like shape
                region = (x / w, y / h, (x + cw) / w, (y + ch) / h)
                calibration['green_regions'].append({
                    'region': region,
                    'area': area,
                    'aspect_ratio': cw / ch if ch > 0 else 0
                })

        # Analyze cyan contours (potential Shield bars)
        for cnt in cyan_contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area > 100 and cw > ch * 2:  # Bar-like shape
                region = (x / w, y / h, (x + cw) / w, (y + ch) / h)
                calibration['cyan_regions'].append({
                    'region': region,
                    'area': area,
                    'aspect_ratio': cw / ch if ch > 0 else 0
                })

        # Select best candidates (largest bar-shaped regions in bottom-right)
        for regions, key in [(calibration['green_regions'], 'recommended_hp_region'),
                            (calibration['cyan_regions'], 'recommended_shield_region')]:
            bottom_right_regions = [r for r in regions
                                   if r['region'][0] > 0.5 and r['region'][1] > 0.5]
            if bottom_right_regions:
                best = max(bottom_right_regions, key=lambda r: r['area'])
                calibration[key] = best['region']

        return calibration

    def get_stats(self) -> Dict:
        """Get reader statistics."""
        return {
            'ocr_backend': self.config.ocr_backend,
            'last_hp': self._smoothed_hp,
            'last_shield': self._smoothed_shield,
            'last_read_time': self._last_read_time,
            'ocr_available': self.ocr_engine is not None
        }

    def reset(self):
        """Reset smoothed values and cache."""
        self._smoothed_hp = 1.0
        self._smoothed_shield = 1.0
        self._last_values = None
        self._last_read_time = 0


class TargetHPReader:
    """
    Reads HP bars of targeted enemies/objects.

    Ships in DarkOrbit have HP/Shield bars displayed above them.
    This class detects and reads those bars.
    """

    def __init__(self):
        # Bar detection settings
        self.hp_bar_colors = {
            'green': ((35, 100, 100), (85, 255, 255)),
            'yellow': ((20, 100, 100), (35, 255, 255)),
            'red': ((0, 100, 100), (10, 255, 255))
        }
        self.shield_bar_color = ((85, 100, 100), (130, 255, 255))

        # Cache
        self._last_target_hp: Dict[int, float] = {}  # track_id -> hp

    def read_target_hp(self, frame: np.ndarray,
                       bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Read HP and Shield of a target from the area above its bounding box.

        Args:
            frame: BGR image
            bbox: Target bounding box (x_center, y_center, width, height) normalized

        Returns:
            (hp_ratio, shield_ratio) - both 0.0 to 1.0
        """
        h, w = frame.shape[:2]

        # Convert normalized bbox to pixels
        cx = int(bbox[0] * w)
        cy = int(bbox[1] * h)
        bw = int(bbox[2] * w)
        bh = int(bbox[3] * h)

        # HP bar is typically above the target
        bar_height = max(10, int(bh * 0.15))
        bar_width = max(30, int(bw * 1.2))

        # Region above the target
        x1 = max(0, cx - bar_width // 2)
        x2 = min(w, cx + bar_width // 2)
        y1 = max(0, cy - bh // 2 - bar_height * 2)
        y2 = max(0, cy - bh // 2)

        if y1 >= y2 or x1 >= x2:
            return 1.0, 1.0

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0, 1.0

        # Detect HP bar (look for green/yellow/red horizontal bar)
        hp_ratio = self._detect_hp_bar(region)

        # Detect Shield bar (cyan, usually above HP bar)
        shield_ratio = self._detect_shield_bar(region)

        return hp_ratio, shield_ratio

    def _detect_hp_bar(self, region: np.ndarray) -> float:
        """Detect HP bar fill level."""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        total_mask = np.zeros(region.shape[:2], dtype=np.uint8)

        for color_name, (low, high) in self.hp_bar_colors.items():
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            total_mask = cv2.bitwise_or(total_mask, mask)

        # Measure horizontal fill
        h, w = total_mask.shape
        if w == 0:
            return 1.0

        col_sums = np.sum(total_mask > 0, axis=0)
        threshold = h * 0.2
        filled_cols = np.sum(col_sums > threshold)

        return filled_cols / w if w > 0 else 1.0

    def _detect_shield_bar(self, region: np.ndarray) -> float:
        """Detect Shield bar fill level."""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        low, high = self.shield_bar_color
        mask = cv2.inRange(hsv, np.array(low), np.array(high))

        h, w = mask.shape
        if w == 0:
            return 1.0

        col_sums = np.sum(mask > 0, axis=0)
        threshold = h * 0.2
        filled_cols = np.sum(col_sums > threshold)

        return filled_cols / w if w > 0 else 1.0


# Convenience function
def create_hud_reader(ocr_backend: str = "easyocr", **kwargs) -> HUDReader:
    """
    Create a HUD reader with specified backend.

    Args:
        ocr_backend: "easyocr", "tesseract", "paddleocr", or "color_only"
        **kwargs: Additional config options

    Returns:
        Configured HUDReader instance
    """
    config = HUDConfig(ocr_backend=ocr_backend, **kwargs)
    return HUDReader(config)
