"""
GUI Detection and Masking System (YOLO-based)

Uses YOLO detections to mask GUI elements and prevent unwanted clicks.

GUI classes detected by YOLO:
- Minimap (for position extraction)
- ChatWindow
- HotbarWindow/Hotbar
- MenuBar
- Sidebar_menu
- LogWindow
- GroupWindow, MissionsWindow, ShipWindow, UserWindow
- UserInfo (HP/shield bars)
- AmmoDisplay
- Button, Portal_btn

For the minimap specifically:
1. YOLO provides bounding box
2. OpenCV analyzes inside to find player position (brightest pixel)

No hardcoded regions - fully dynamic based on YOLO detections!
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GUIRegion:
    """Represents a GUI element region that should not be clicked."""
    name: str
    x: float  # Normalized 0-1
    y: float
    width: float
    height: float
    mask_clicks: bool = True  # Whether to prevent clicks in this region

    def contains(self, x: float, y: float) -> bool:
        """Check if normalized coordinates are inside this region."""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def to_absolute(self, screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates (x1, y1, x2, y2)."""
        x1 = int(self.x * screen_width)
        y1 = int(self.y * screen_height)
        x2 = int((self.x + self.width) * screen_width)
        y2 = int((self.y + self.height) * screen_height)
        return (x1, y1, x2, y2)


class MinimapDetector:
    """
    Extracts strategic information from minimap using OpenCV.

    The minimap bounding box is provided by YOLO detection.
    This class analyzes the pixels INSIDE the minimap to extract:
    - Player position (white/yellow/green marker)
    - Enemy positions (red dots)
    - Ally positions (blue dots)
    - Portal locations (yellow/green markers)
    - Map boundaries (detected from minimap border)

    All positions are normalized (0-1) on the actual game map.
    """

    def __init__(self):
        """Initialize minimap analyzer."""
        # Cached minimap location (provided by YOLO)
        self.minimap_region: Optional[GUIRegion] = None

        # Player position on minimap (normalized within game map)
        self.player_map_x = 0.5
        self.player_map_y = 0.5

        # Enemy positions (list of (x, y) tuples, normalized 0-1)
        self.enemy_positions: List[Tuple[float, float]] = []

        # Ally positions (list of (x, y) tuples, normalized 0-1)
        self.ally_positions: List[Tuple[float, float]] = []

        # Portal positions (list of (x, y) tuples, normalized 0-1)
        self.portal_positions: List[Tuple[float, float]] = []

    def set_minimap_region(self, region: GUIRegion):
        """
        Set minimap region from YOLO detection.

        Args:
            region: GUIRegion for minimap from YOLO
        """
        self.minimap_region = region

    def analyze_minimap(self, frame: np.ndarray) -> Dict:
        """
        Comprehensive minimap analysis - extracts ALL strategic information.

        Analyzes colors to find:
        - Player position (brightest/white/yellow marker)
        - Enemies (red dots)
        - Allies (blue dots)
        - Portals (green/yellow markers)

        Args:
            frame: BGR frame

        Returns:
            Dict with all minimap data
        """
        if self.minimap_region is None:
            return {
                'player_pos': (0.5, 0.5),
                'enemies': [],
                'allies': [],
                'portals': []
            }

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.minimap_region.to_absolute(w, h)

        # Extract minimap ROI
        minimap_roi = frame[y1:y2, x1:x2]

        if minimap_roi.size == 0:
            return {'player_pos': (0.5, 0.5), 'enemies': [], 'allies': [], 'portals': []}

        map_h, map_w = minimap_roi.shape[:2]

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(minimap_roi, cv2.COLOR_BGR2HSV)

        # 1. Find PLAYER (brightest point or green/yellow)
        gray = cv2.cvtColor(minimap_roi, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        player_pos = (0.5, 0.5)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                player_pos = (cx / map_w, cy / map_h)

        self.player_map_x, self.player_map_y = player_pos

        # 2. Find ENEMIES (red dots)
        # Red in HSV: Hue 0-10 or 170-180
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        enemy_positions = []
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 2:  # Filter noise
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    enemy_positions.append((cx / map_w, cy / map_h))

        self.enemy_positions = enemy_positions

        # 3. Find ALLIES (blue dots)
        # Blue in HSV: Hue 100-130
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        ally_positions = []
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 2:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    ally_positions.append((cx / map_w, cy / map_h))

        self.ally_positions = ally_positions

        # 4. Find PORTALS (white circles, larger than dots)
        # Use the bright_mask from player detection but filter by size
        portal_positions = []

        # Find circles in bright areas (portals are circular)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)

            # Portals are bigger than dots but smaller than player marker
            # Typical sizes: dots ~2-5 pixels, portals ~10-30 pixels, player ~40+ pixels
            if 10 < area < 100:  # Adjust these thresholds as needed
                # Check if it's roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # Circles have circularity close to 1.0
                    if circularity > 0.7:  # Fairly circular
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]

                            # Make sure it's not the player marker
                            dist_to_player = np.sqrt((cx/map_w - player_pos[0])**2 +
                                                    (cy/map_h - player_pos[1])**2)
                            if dist_to_player > 0.05:  # Not too close to player
                                portal_positions.append((cx / map_w, cy / map_h))

        self.portal_positions = portal_positions

        return {
            'player_pos': player_pos,
            'enemies': enemy_positions,
            'allies': ally_positions,
            'portals': portal_positions
        }

    def extract_player_position(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Extract just player position (lightweight version).

        For backward compatibility - use analyze_minimap() for full analysis.

        Returns:
            (map_x, map_y) normalized position on the actual game map (0-1)
        """
        # Run full analysis (caches results)
        result = self.analyze_minimap(frame)
        return result['player_pos']


class FastLogReader:
    """
    Reads combat log messages from TOP SCREEN fast logs using OCR.

    DarkOrbit shows recent combat events at the TOP CENTER of screen:
    - "You destroyed X" - Combat success
    - "You received X damage" - Taking damage
    - "Base attacked!" - Emergency events
    - "X player entered map" - PvP warning

    These fade after a few seconds, so we read them frequently.
    Much faster than opening log window!

    OCR runs in background thread to avoid blocking main loop.
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """Initialize fast log reader."""
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Define fast log region (top center of screen)
        # DarkOrbit fast logs appear at VERY TOP CENTER (top 1-2% of screen)
        # Based on screenshot: logs appear right below the top UI bar
        self.log_region = GUIRegion(
            name="fast_logs",
            x=0.25,      # Start at 25% from left (centered)
            y=0.005,     # Start at 0.5% from top (VERY top of screen)
            width=0.50,  # 50% width (wide enough to catch multi-line logs)
            height=0.04, # 4% height (VERY small - just the log notification area)
            mask_clicks=False  # Don't block clicks here
        )

        self.recent_logs: List[str] = []
        self.recent_logs_with_time: deque = deque(maxlen=10)  # (message, timestamp)
        self.last_read_time = 0.0
        self.last_submit_time = 0.0  # Track when we last submitted a frame (separate from OCR completion)
        self.read_interval = 0.1  # Read every 100ms for fast response!

        # Event tracking (persistent across frames)
        self.recent_events = {
            'kills': 0,
            'damage_taken': 0,
            'rewards': 0,
            'alerts': 0
        }
        self.event_decay_time = 5.0  # Events decay after 5 seconds
        self.event_timestamps: List[Tuple[str, float]] = []  # (event_type, timestamp)

        # Background OCR thread
        import threading
        import queue
        self.ocr_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.ocr_thread = None
        self.ocr_running = False
        self.ocr_lock = threading.Lock()  # Protect access to recent_logs

        # Try to import OCR backend (prefer RapidOCR for speed + accuracy)
        self.ocr_backend = None
        self.ocr_available = False

        try:
            from rapidocr_onnxruntime import RapidOCR
            # Try to use GPU if available (DirectML for Windows, CUDA for Linux)
            try:
                self.reader = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
                print("[OCR-INIT] RapidOCR initialized with CUDA GPU acceleration")
                logger.info("RapidOCR initialized with CUDA GPU acceleration")
            except Exception as e:
                # Fallback to CPU if GPU not available
                self.reader = RapidOCR()
                print(f"[OCR-INIT] RapidOCR running on CPU (GPU error: {e}) - may be slow!")
                logger.warning(f"RapidOCR running on CPU (no CUDA available): {e}")
            self.ocr_backend = 'rapidocr'
            self.ocr_available = True
        except ImportError:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                self.ocr_backend = 'easyocr'
                self.ocr_available = True
                print("[OCR-INIT] EasyOCR initialized for fast log reading (GPU-accelerated)")
                logger.info("EasyOCR initialized for fast log reading (GPU-accelerated)")
            except ImportError:
                try:
                    import pytesseract
                    self.ocr_backend = 'pytesseract'
                    self.ocr_available = True
                    print("[OCR-INIT] pytesseract initialized for fast log reading")
                    logger.info("pytesseract initialized for fast log reading")
                except ImportError:
                    print("[OCR-INIT] No OCR backend available - fast log reading disabled")
                    logger.warning("No OCR backend available (install rapidocr-onnxruntime, easyocr or pytesseract), fast log reading disabled")
                    self.ocr_available = False

        # Start background OCR thread if OCR is available
        if self.ocr_available:
            self._start_ocr_thread()

    def _start_ocr_thread(self):
        """Start background OCR processing thread."""
        import threading
        self.ocr_running = True
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread.start()
        print("[OCR-THREAD] Background OCR thread started")

    def _ocr_worker(self):
        """Background OCR worker thread - processes frames from queue."""
        import queue

        if not hasattr(self, '_ocr_timing_counter'):
            self._ocr_timing_counter = 0

        while self.ocr_running:
            try:
                # Get latest frame from queue (non-blocking with timeout)
                try:
                    log_roi, current_time = self.ocr_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process OCR (this is the slow part that now runs in background)
                lines = []
                ocr_start = time.time()

                if self.ocr_backend == 'rapidocr':
                    # RapidOCR approach - ONNX-based, fast and accurate
                    gray = cv2.cvtColor(log_roi, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)

                    result, _ = self.reader(enhanced)

                    if result:
                        lines = [item[1].strip() for item in result if item[1].strip()]

                elif self.ocr_backend == 'easyocr':
                    gray = cv2.cvtColor(log_roi, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    result = self.reader.readtext(enhanced, detail=0, paragraph=False)
                    lines = [line.strip() for line in result if line.strip()]

                else:  # pytesseract
                    import pytesseract

                    gray = cv2.cvtColor(log_roi, cv2.COLOR_BGR2GRAY)
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )

                    if np.mean(binary) > 127:
                        binary = cv2.bitwise_not(binary)

                    text = pytesseract.image_to_string(binary, config='--psm 6')
                    lines = [line.strip() for line in text.split('\n') if line.strip()]

                # Update results (thread-safe)
                with self.ocr_lock:
                    if lines != self.recent_logs:
                        # New messages detected!
                        self._parse_new_events(lines, current_time)
                        # Add new logs with timestamps (keep last 10)
                        for line in lines:
                            if line not in self.recent_logs:
                                self.recent_logs_with_time.append((line, current_time))

                    self.recent_logs = lines
                    self.last_read_time = current_time

                # Log OCR timing
                ocr_time = (time.time() - ocr_start) * 1000
                self._ocr_timing_counter += 1

                if self._ocr_timing_counter % 10 == 0:
                    print(f"   [OCR-TIMING] {self.ocr_backend} took {ocr_time:.1f}ms | Read {len(lines)} lines | Lines: {lines}")

            except Exception as e:
                print(f"[OCR-THREAD] Error in OCR worker: {e}")
                import traceback
                traceback.print_exc()

    def read_logs(self, frame: np.ndarray, force: bool = False) -> List[str]:
        """
        Read recent log messages from top screen fast logs.

        This now submits frames to background OCR thread and returns cached results.
        Non-blocking - OCR runs in background, main loop stays fast!

        Args:
            frame: BGR frame
            force: Force submit even if interval not elapsed

        Returns:
            List of recent log messages (from last OCR result)
        """
        if not self.ocr_available:
            return []

        # Rate limit frame submission (based on SUBMIT time, not OCR completion time)
        current_time = time.time()
        time_since_submit = current_time - self.last_submit_time

        # Submit frame to background OCR if interval elapsed
        if force or time_since_submit >= self.read_interval:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self.log_region.to_absolute(w, h)

            # Extract fast log ROI (top center of screen)
            log_roi = frame[y1:y2, x1:x2]

            if log_roi.size > 0:
                # Submit to background OCR thread (non-blocking!)
                # Queue size is 1, so if OCR is still processing, this replaces the old frame
                try:
                    # Copy the ROI to avoid race conditions
                    log_roi_copy = log_roi.copy()

                    # Try to put in queue without blocking
                    # If queue is full, drain it first (discard old frame)
                    if self.ocr_queue.full():
                        try:
                            self.ocr_queue.get_nowait()
                        except:
                            pass

                    self.ocr_queue.put_nowait((log_roi_copy, current_time))
                    self.last_submit_time = current_time  # Track submission time
                except:
                    # Queue operations failed, just skip this frame
                    pass

        # Return cached results (thread-safe read)
        with self.ocr_lock:
            return self.recent_logs.copy()

    def _parse_new_events(self, new_lines: List[str], timestamp: float):
        """Parse new log lines for events and update tracking."""
        for line in new_lines:
            if line in self.recent_logs:
                continue  # Already processed

            line_lower = line.lower()
            event_type = None

            if 'destroyed' in line_lower or 'killed' in line_lower:
                event_type = 'kills'
            elif 'received' in line_lower and 'damage' in line_lower:
                event_type = 'damage_taken'
            elif 'collected' in line_lower or 'bonus' in line_lower:
                event_type = 'rewards'
            elif 'attacked' in line_lower or 'warning' in line_lower or 'entered' in line_lower:
                event_type = 'alerts'

            if event_type:
                self.event_timestamps.append((event_type, timestamp))

    def get_recent_events(self) -> Dict[str, int]:
        """
        Get event counts from last 5 seconds (with decay).

        Returns:
            Dict with event counts: {
                'kills': int,
                'damage_taken': int,
                'rewards': int,
                'alerts': int
            }
        """
        current_time = time.time()

        # Remove old events (older than decay time)
        self.event_timestamps = [
            (event_type, ts) for event_type, ts in self.event_timestamps
            if current_time - ts < self.event_decay_time
        ]

        # Count recent events
        events = {
            'kills': 0,
            'damage_taken': 0,
            'rewards': 0,
            'alerts': 0
        }

        for event_type, ts in self.event_timestamps:
            if event_type in events:
                events[event_type] += 1

        return events

    def stop(self):
        """Stop the background OCR thread."""
        if self.ocr_thread and self.ocr_running:
            print("[OCR-THREAD] Stopping background OCR thread...")
            self.ocr_running = False
            if self.ocr_thread.is_alive():
                self.ocr_thread.join(timeout=2.0)
            print("[OCR-THREAD] Background OCR thread stopped")


class GUIDetector:
    """
    Main GUI detection system using YOLO detections.

    No hardcoded regions - fully dynamic based on YOLO!
    """

    # GUI classes to mask (from YOLO)
    GUI_CLASSES = {
        'Minimap', 'ChatWindow', 'HotbarWindow', 'Hotbar',
        'MenuBar', 'Sidebar_menu', 'LogWindow', 'GroupWindow',
        'MissionsWindow', 'ShipWindow', 'UserWindow', 'UserInfo',
        'AmmoDisplay', 'Button', 'Portal_btn'
    }

    def __init__(self,
                 screen_width: int = 1920,
                 screen_height: int = 1080,
                 enable_log_reading: bool = True):
        """
        Args:
            screen_width: Screen resolution width
            screen_height: Screen resolution height
            enable_log_reading: Enable fast log reading (top screen)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # GUI regions from YOLO detections
        self.gui_regions: List[GUIRegion] = []

        # Minimap analyzer (player, enemies, allies, portals)
        self.minimap_detector = MinimapDetector()

        # Fast log reader (reads top screen combat notifications)
        self.log_reader = FastLogReader(screen_width, screen_height) if enable_log_reading else None

    def update_from_detections(self, detections: List) -> None:
        """
        Update GUI regions from YOLO detections.

        Args:
            detections: List of Detection objects from YOLO
        """
        self.gui_regions = []

        for det in detections:
            class_name = det.class_name

            # Check if this is a GUI element
            if class_name in self.GUI_CLASSES:
                # Detection format: x_center, y_center, width, height (all normalized 0-1)
                # Convert to top-left corner format for GUIRegion
                norm_x = det.x_center - (det.width / 2)
                norm_y = det.y_center - (det.height / 2)

                region = GUIRegion(
                    name=class_name,
                    x=norm_x,
                    y=norm_y,
                    width=det.width,
                    height=det.height,
                    mask_clicks=True
                )

                self.gui_regions.append(region)

                # Special handling for minimap
                if class_name == 'Minimap':
                    self.minimap_detector.set_minimap_region(region)

    def is_click_allowed(self, x: float, y: float) -> bool:
        """
        Check if a click is allowed at normalized coordinates.

        Args:
            x: Normalized x (0-1)
            y: Normalized y (0-1)

        Returns:
            True if click is allowed, False if blocked by GUI
        """
        # Check all GUI regions from YOLO
        for region in self.gui_regions:
            if region.mask_clicks and region.contains(x, y):
                return False

        return True

    def get_safe_click_position(self,
                                 target_x: float,
                                 target_y: float,
                                 max_search_radius: float = 0.1) -> Tuple[float, float]:
        """
        Find nearest safe click position to target.

        If target is inside GUI, find nearest allowed position.

        Args:
            target_x: Desired normalized x
            target_y: Desired normalized y
            max_search_radius: Maximum search radius (normalized)

        Returns:
            (safe_x, safe_y) nearest safe position, or original if already safe
        """
        if self.is_click_allowed(target_x, target_y):
            return (target_x, target_y)

        # Spiral search for nearest safe position
        step = 0.01  # 1% of screen
        radius = step

        while radius <= max_search_radius:
            # Sample points in a circle
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                test_x = target_x + radius * np.cos(angle)
                test_y = target_y + radius * np.sin(angle)

                # Clamp to screen bounds
                test_x = np.clip(test_x, 0.0, 1.0)
                test_y = np.clip(test_y, 0.0, 1.0)

                if self.is_click_allowed(test_x, test_y):
                    return (test_x, test_y)

            radius += step

        # No safe position found, return original (with warning)
        logger.warning(f"No safe click position found near ({target_x:.2f}, {target_y:.2f})")
        return (target_x, target_y)

    def get_map_position(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Get player's position on the actual game map.

        Args:
            frame: BGR frame

        Returns:
            (map_x, map_y) normalized position (0-1)
        """
        return self.minimap_detector.extract_player_position(frame)

    def get_full_minimap_data(self, frame: np.ndarray) -> Dict:
        """
        Get comprehensive minimap analysis.

        Returns:
            Dict with:
                'player_pos': (x, y) tuple
                'enemies': List[(x, y), ...]
                'allies': List[(x, y), ...]
                'portals': List[(x, y), ...]
        """
        return self.minimap_detector.analyze_minimap(frame)

    def get_log_events(self, frame: np.ndarray) -> Dict[str, int]:
        """
        Get recent combat log events.

        Returns:
            Dict with event counts: {
                'kills': int,
                'damage_taken': int,
                'rewards': int,
                'alerts': int
            }
        """
        if self.log_reader is None:
            return {'kills': 0, 'damage_taken': 0, 'rewards': 0, 'alerts': 0}

        # Read logs (rate-limited internally)
        self.log_reader.read_logs(frame)

        # Parse for events
        return self.log_reader.get_recent_events()

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize GUI regions on frame (for debugging).

        Args:
            frame: BGR frame

        Returns:
            Frame with GUI regions drawn
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw all YOLO-detected GUI regions in cyan
        for region in self.gui_regions:
            x1, y1, x2, y2 = region.to_absolute(w, h)

            # Use different color for minimap
            color = (255, 255, 0) if region.name == 'Minimap' else (255, 255, 0)  # Cyan for all

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, region.name, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw minimap analysis (player, enemies, allies, portals)
        if self.minimap_detector.minimap_region:
            minimap_region = self.minimap_detector.minimap_region
            mx1, my1, mx2, my2 = minimap_region.to_absolute(w, h)

            # Helper to convert normalized minimap coords to screen pixels
            def map_to_screen(map_x, map_y):
                px = int(mx1 + map_x * (mx2 - mx1))
                py = int(my1 + map_y * (my2 - my1))
                return (px, py)

            # Draw enemies (red dots)
            for enemy_x, enemy_y in self.minimap_detector.enemy_positions:
                pos = map_to_screen(enemy_x, enemy_y)
                cv2.circle(overlay, pos, 3, (0, 0, 255), -1)  # Red

            # Draw allies (blue dots)
            for ally_x, ally_y in self.minimap_detector.ally_positions:
                pos = map_to_screen(ally_x, ally_y)
                cv2.circle(overlay, pos, 3, (255, 0, 0), -1)  # Blue

            # Draw portals (green markers)
            for portal_x, portal_y in self.minimap_detector.portal_positions:
                pos = map_to_screen(portal_x, portal_y)
                cv2.circle(overlay, pos, 4, (0, 255, 255), 2)  # Cyan ring

            # Draw player position (green dot, larger)
            map_x, map_y = self.minimap_detector.player_map_x, self.minimap_detector.player_map_y
            player_pos = map_to_screen(map_x, map_y)
            cv2.circle(overlay, player_pos, 5, (0, 255, 0), -1)  # Green filled

        # Draw log events if available
        if self.log_reader and self.log_reader.recent_logs:
            y_offset = h - 150
            cv2.putText(overlay, "Recent Logs:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

            for log in self.log_reader.recent_logs[-3:]:  # Show last 3
                cv2.putText(overlay, log[:50], (10, y_offset),  # Truncate long logs
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 15

        return overlay
