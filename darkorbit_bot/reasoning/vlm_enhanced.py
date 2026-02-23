"""
Enhanced VLM Analysis System

Multi-level analysis inspired by V2's hierarchical architecture:
- STRATEGIC: Long-term situation assessment (every 5-10s)
- TACTICAL: Target prioritization and tactics (every 1-2s)
- EXECUTION: Immediate action validation (every 0.3-0.5s)

Features:
- Memory accumulation across queries
- Specialized prompts per situation
- Motion/change detection
- Multi-frame temporal analysis
"""

import json
import time
import base64
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from io import BytesIO

try:
    import requests
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    requests = None
    np = None
    Image = None


@dataclass
class VLMMemory:
    """Accumulated knowledge from VLM sessions."""
    # Recent observations
    recent_enemies: List[str] = field(default_factory=list)  # Last seen enemy types
    recent_tactics: List[str] = field(default_factory=list)  # Detected tactics
    recent_issues: List[str] = field(default_factory=list)   # Problems found

    # Session stats
    total_critiques: int = 0
    good_actions: int = 0
    bad_actions: int = 0

    # Danger zones (areas where bot took damage or died)
    danger_positions: List[Tuple[float, float]] = field(default_factory=list)

    # Strategic context (updated by strategic VLM)
    current_strategy: str = "unknown"  # farming/hunting/fleeing/exploring
    threat_level: str = "unknown"
    area_safety: str = "unknown"

    # Last analysis timestamps
    last_strategic: float = 0
    last_tactical: float = 0
    last_execution: float = 0

    def add_observation(self, enemy_type: str = None, tactic: str = None, issue: str = None):
        """Add observation to memory (keeps last 10 of each)."""
        if enemy_type:
            self.recent_enemies.append(enemy_type)
            self.recent_enemies = self.recent_enemies[-10:]
        if tactic:
            self.recent_tactics.append(tactic)
            self.recent_tactics = self.recent_tactics[-10:]
        if issue:
            self.recent_issues.append(issue)
            self.recent_issues = self.recent_issues[-10:]

    def get_context_summary(self) -> str:
        """Get summary string for VLM context."""
        lines = []
        if self.recent_enemies:
            enemy_counts = {}
            for e in self.recent_enemies:
                enemy_counts[e] = enemy_counts.get(e, 0) + 1
            lines.append(f"Recent enemies: {enemy_counts}")
        if self.recent_tactics:
            lines.append(f"Recent tactics used: {set(self.recent_tactics)}")
        if self.recent_issues:
            lines.append(f"Recent issues: {self.recent_issues[-3:]}")
        lines.append(f"Session: {self.good_actions} good, {self.bad_actions} bad actions")
        lines.append(f"Strategy: {self.current_strategy}, Threat: {self.threat_level}")
        return "\n".join(lines) if lines else "(no prior context)"


class EnhancedVLM:
    """
    Multi-level VLM analysis system.

    Runs three levels of analysis at different frequencies:
    - Strategic (5-10s): Overall situation, area safety, long-term goals
    - Tactical (1-2s): Target selection, tactic choice, threat assessment
    - Execution (0.3-0.5s): Immediate action validation
    """

    # System prompt file path
    SYSTEM_PROMPT_FILE = Path(__file__).parent.parent / "data" / "vlm_system_prompt.txt"

    # Analysis intervals (seconds)
    STRATEGIC_INTERVAL = 5.0
    TACTICAL_INTERVAL = 1.0
    EXECUTION_INTERVAL = 0.3

    # Specialized prompts for each level
    STRATEGIC_PROMPT = """=== STRATEGIC ANALYSIS ===

You are analyzing the OVERALL SITUATION in a DarkOrbit game session.

PRIOR CONTEXT (from previous analyses):
{memory_context}

CURRENT FRAME shows the game state.

Analyze:
1. AREA ASSESSMENT: Is this a safe farming area or dangerous zone?
2. RESOURCE STATUS: Does player seem healthy? Are there good targets?
3. STRATEGIC RECOMMENDATION: What should the bot's overall goal be?
   - FARM: Stay and kill enemies for resources
   - HUNT: Actively seek stronger enemies
   - FLEE: Area is too dangerous, retreat
   - EXPLORE: Move to find better targets
   - LOOT: Focus on collecting boxes

Reply with JSON:
{{
  "area_type": "safe_farming/contested/dangerous/empty",
  "threat_assessment": "none/low/medium/high/critical",
  "resource_status": "healthy/moderate/low/critical",
  "recommended_strategy": "farm/hunt/flee/explore/loot",
  "reasoning": "brief explanation",
  "stay_duration": "how long bot should stay in this area (seconds)"
}}"""

    TACTICAL_PROMPT = """=== TACTICAL ANALYSIS ===

You are analyzing TARGET SELECTION and COMBAT TACTICS.

STRATEGIC CONTEXT:
- Current strategy: {strategy}
- Area threat level: {threat_level}
- Prior context: {memory_context}

CURRENT FRAME shows detected enemies (red boxes) and loot (yellow boxes).

Analyze:
1. TARGET PRIORITY: Which target should be attacked first and why?
2. TACTIC SELECTION: What combat tactic should be used?
   - ORBIT: Circle around enemy at safe distance
   - KITE: Move away while attacking
   - RUSH: Close distance quickly (weak enemies only)
   - RETREAT: Disengage and flee
3. DANGER CHECK: Any threats the bot should avoid?

Reply with JSON:
{{
  "priority_target": {{
    "type": "enemy/box/none",
    "name": "enemy class name or 'BonusBox'",
    "reason": "why this target"
  }},
  "recommended_tactic": "orbit/kite/rush/retreat/collect",
  "tactic_params": {{
    "distance": 0.1,
    "direction": "clockwise/counterclockwise/away/toward"
  }},
  "threats_detected": ["list of dangerous things to avoid"],
  "confidence": 0.8
}}"""

    EXECUTION_PROMPT = """=== EXECUTION CHECK ===

Quick validation of the bot's CURRENT ACTION.

CONTEXT:
- Target: {current_target}
- Tactic: {current_tactic}
- Bot position: {bot_position}
- Bot attacking: {is_attacking}

Was the bot's last action CORRECT for this situation?

Reply with JSON:
{{
  "action_correct": true,
  "issue": "none or brief description of problem",
  "correction": {{
    "move_x": 0.5,
    "move_y": 0.5,
    "should_attack": true
  }}
}}"""

    MOTION_ANALYSIS_PROMPT = """=== MOTION ANALYSIS ===

You are analyzing MOVEMENT PATTERNS across {num_frames} frames over {duration:.1f} seconds.

The image shows a sequence of frames with motion indicators:
- Arrows show direction of movement
- Colored trails show path taken

PRIOR TACTICAL CONTEXT:
- Expected tactic: {expected_tactic}
- Target type: {target_type}

Analyze the MOVEMENT QUALITY:
1. Is the bot executing the expected tactic correctly?
2. Is the movement smooth or erratic?
3. Is the bot maintaining appropriate distance?

Reply with JSON:
{{
  "tactic_detected": "orbit/kite/rush/stationary/erratic/none",
  "tactic_quality": "good/acceptable/poor",
  "movement_smoothness": "smooth/jerky/erratic",
  "distance_management": "too_close/optimal/too_far/inconsistent",
  "specific_issues": ["list of movement problems"],
  "improvement": "what should change"
}}"""

    def __init__(self,
                 base_url: str = "http://localhost:1234",
                 model: str = "local-model",
                 fast_model: str = None):  # Optional faster model for execution checks
        """
        Initialize enhanced VLM.

        Args:
            base_url: LM Studio URL
            model: Primary model for tactical/strategic
            fast_model: Optional faster model for execution checks (uses primary if None)
        """
        self.base_url = base_url
        self.model = model
        self.fast_model = fast_model or model

        # Memory
        self.memory = VLMMemory()

        # Frame buffer for motion analysis
        self.frame_buffer: deque = deque(maxlen=10)

        # Current tactical state (from last tactical analysis)
        self.current_target = None
        self.current_tactic = None

        # Threading for parallel queries
        self._lock = threading.Lock()

        # Load system prompt
        self._system_prompt = self._load_system_prompt()

        # Correction saving (for training data generation)
        self._corrections_enabled = False
        self._corrections = []
        self._correction_count = 0

        # Async processing - don't block main control loop!
        self._running = False
        self._analysis_thread: Optional[threading.Thread] = None
        self._pending_frame = None
        self._pending_detections = None
        self._pending_action = None
        self._pending_context = None
        self._last_results = {
            'strategic': None,
            'tactical': None,
            'execution': None,
            'motion': None
        }

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            if self.SYSTEM_PROMPT_FILE.exists():
                return self.SYSTEM_PROMPT_FILE.read_text(encoding='utf-8')
        except Exception:
            pass
        return "You are an expert DarkOrbit game analyst."

    def add_frame(self, frame: np.ndarray, detections: List, context: Dict):
        """
        Add frame to buffer for temporal analysis.

        Call this every frame to build motion history.
        """
        self.frame_buffer.append({
            'frame': frame.copy() if frame is not None else None,
            'detections': detections,
            'context': context,
            'time': time.time()
        })

    def should_run_strategic(self) -> bool:
        """Check if strategic analysis is due."""
        return time.time() - self.memory.last_strategic > self.STRATEGIC_INTERVAL

    def should_run_tactical(self) -> bool:
        """Check if tactical analysis is due."""
        return time.time() - self.memory.last_tactical > self.TACTICAL_INTERVAL

    def should_run_execution(self) -> bool:
        """Check if execution check is due."""
        return time.time() - self.memory.last_execution > self.EXECUTION_INTERVAL

    def analyze_strategic(self, frame: np.ndarray, detections: List,
                           bot_action: Dict = None, context: Dict = None) -> Optional[Dict]:
        """
        Run strategic analysis - overall situation assessment.

        Call every 5-10 seconds.
        """
        if frame is None:
            return None

        self.memory.last_strategic = time.time()

        prompt = self.STRATEGIC_PROMPT.format(
            memory_context=self.memory.get_context_summary()
        )

        result = self._query_vlm(frame, detections, prompt, use_fast=False)

        if result:
            # Update memory with strategic context
            self.memory.current_strategy = result.get('recommended_strategy', 'unknown')
            self.memory.threat_level = result.get('threat_assessment', 'unknown')
            self.memory.area_safety = result.get('area_type', 'unknown')

            # Save correction for training if enabled
            if bot_action and context:
                self.save_correction(bot_action, result, detections, context, level="strategic")

        return result

    def analyze_tactical(self, frame: np.ndarray, detections: List,
                          bot_action: Dict = None, context: Dict = None) -> Optional[Dict]:
        """
        Run tactical analysis - target and tactic selection.

        Call every 1-2 seconds.
        """
        if frame is None:
            return None

        self.memory.last_tactical = time.time()

        prompt = self.TACTICAL_PROMPT.format(
            strategy=self.memory.current_strategy,
            threat_level=self.memory.threat_level,
            memory_context=self.memory.get_context_summary()
        )

        result = self._query_vlm(frame, detections, prompt, use_fast=False)

        if result:
            # Update current tactical state
            target = result.get('priority_target', {})
            self.current_target = target.get('name')
            self.current_tactic = result.get('recommended_tactic')

            # Update memory
            if target.get('type') == 'enemy':
                self.memory.add_observation(enemy_type=target.get('name'))
            self.memory.add_observation(tactic=self.current_tactic)

            # Save correction for training if enabled
            if bot_action and context:
                self.save_correction(bot_action, result, detections, context, level="tactical")

        return result

    def analyze_execution(self, frame: np.ndarray, detections: List,
                          bot_action: Dict, is_attacking: bool,
                          context: Dict = None) -> Optional[Dict]:
        """
        Quick execution check - was the action correct?

        Call every 0.3-0.5 seconds. Uses fast model if available.
        """
        if frame is None:
            return None

        self.memory.last_execution = time.time()

        # Get bot position from action
        bot_x = bot_action.get('move_x', bot_action.get('aim_x', 0.5))
        bot_y = bot_action.get('move_y', bot_action.get('aim_y', 0.5))

        prompt = self.EXECUTION_PROMPT.format(
            current_target=self.current_target or "unknown",
            current_tactic=self.current_tactic or "unknown",
            bot_position=f"({bot_x:.2f}, {bot_y:.2f})",
            is_attacking=is_attacking
        )

        result = self._query_vlm(frame, detections, prompt, use_fast=True)

        if result:
            # Update memory stats
            self.memory.total_critiques += 1
            if result.get('action_correct', False):
                self.memory.good_actions += 1
            else:
                self.memory.bad_actions += 1
                issue = result.get('issue', 'unknown')
                if issue != 'none':
                    self.memory.add_observation(issue=issue)

                # Save correction for training if action was wrong
                if context:
                    self.save_correction(bot_action, result, detections, context, level="execution")

        return result

    def analyze_motion(self, expected_tactic: str = None,
                       target_type: str = None) -> Optional[Dict]:
        """
        Analyze motion patterns across recent frames.

        Uses frame buffer to detect movement quality.
        """
        if len(self.frame_buffer) < 4:
            return None

        # Create motion visualization
        motion_image = self._create_motion_image()
        if motion_image is None:
            return None

        # Calculate duration
        oldest = self.frame_buffer[0]['time']
        newest = self.frame_buffer[-1]['time']
        duration = newest - oldest

        prompt = self.MOTION_ANALYSIS_PROMPT.format(
            num_frames=len(self.frame_buffer),
            duration=duration,
            expected_tactic=expected_tactic or self.current_tactic or "unknown",
            target_type=target_type or self.current_target or "unknown"
        )

        # Query with motion image (no detections overlay needed)
        result = self._query_vlm_image(motion_image, prompt, use_fast=False)

        if result:
            detected = result.get('tactic_detected', 'none')
            self.memory.add_observation(tactic=detected)

        return result

    def _create_motion_image(self) -> Optional[Image.Image]:
        """
        Create visualization showing motion across frames.

        Returns a composite image with:
        - Recent frames in a strip
        - Motion vectors/trails overlaid
        """
        try:
            frames = list(self.frame_buffer)
            if len(frames) < 4:
                return None

            # Take 4 evenly spaced frames
            indices = [0, len(frames)//3, 2*len(frames)//3, -1]
            selected = [frames[i] for i in indices]

            # Create image strip
            images = []
            positions = []

            for f in selected:
                frame = f['frame']
                if frame is None:
                    continue

                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)

                # Convert BGR to RGB
                import cv2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Resize to smaller size
                img = img.resize((320, 180), Image.Resampling.LANCZOS)
                images.append(img)

                # Track mouse position from context
                ctx = f.get('context', {})
                pos = (ctx.get('mouse_x', 0.5), ctx.get('mouse_y', 0.5))
                positions.append(pos)

            if len(images) < 4:
                return None

            # Create 2x2 grid
            composite = Image.new('RGB', (640, 360))
            composite.paste(images[0], (0, 0))
            composite.paste(images[1], (320, 0))
            composite.paste(images[2], (0, 180))
            composite.paste(images[3], (320, 180))

            # Draw motion trail
            draw = ImageDraw.Draw(composite)

            # Scale positions to composite size
            grid_positions = [
                (0, 0), (320, 0), (0, 180), (320, 180)
            ]

            for i, (pos, offset) in enumerate(zip(positions, grid_positions)):
                x = int(pos[0] * 320) + offset[0]
                y = int(pos[1] * 180) + offset[1]

                # Draw position marker
                color = (0, 255, 0) if i < 3 else (255, 255, 0)
                draw.ellipse([x-5, y-5, x+5, y+5], fill=color)

                # Draw arrow to next position
                if i < len(positions) - 1:
                    next_pos = positions[i+1]
                    next_offset = grid_positions[i+1]
                    nx = int(next_pos[0] * 320) + next_offset[0]
                    ny = int(next_pos[1] * 180) + next_offset[1]
                    draw.line([(x, y), (nx, ny)], fill=(255, 0, 0), width=2)

            # Add frame numbers
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            for i, offset in enumerate(grid_positions):
                label = f"T-{len(frames)-1-indices[i]}" if i < 3 else "NOW"
                draw.text((offset[0]+5, offset[1]+5), label, fill=(255, 255, 255), font=font)

            return composite

        except Exception as e:
            print(f"[EnhancedVLM] Motion image error: {e}")
            return None

    def _query_vlm(self, frame: np.ndarray, detections: List,
                   prompt: str, use_fast: bool = False) -> Optional[Dict]:
        """Query VLM with frame and detections."""
        try:
            # Encode frame with detection boxes
            image_b64 = self._encode_frame(frame, detections)
            if not image_b64:
                return None

            return self._query_vlm_b64(image_b64, prompt, use_fast)

        except Exception as e:
            print(f"[EnhancedVLM] Query error: {e}")
            return None

    def _query_vlm_image(self, image: Image.Image, prompt: str,
                         use_fast: bool = False) -> Optional[Dict]:
        """Query VLM with PIL Image."""
        try:
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return self._query_vlm_b64(image_b64, prompt, use_fast)

        except Exception as e:
            print(f"[EnhancedVLM] Image query error: {e}")
            return None

    def _query_vlm_b64(self, image_b64: str, prompt: str,
                       use_fast: bool = False) -> Optional[Dict]:
        """Query VLM with base64 image."""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            model = self.fast_model if use_fast else self.model

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                "max_tokens": 400,
                "temperature": 0.2  # Lower for more consistent outputs
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            raw_text = result['choices'][0]['message']['content']

            # Parse JSON
            text = raw_text.strip()
            if '```' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                text = text[start:end]

            return json.loads(text)

        except requests.exceptions.Timeout:
            print(f"[EnhancedVLM] Timeout (use_fast={use_fast})")
            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"[EnhancedVLM] Request error: {e}")
            return None

    def _encode_frame(self, frame: np.ndarray, detections: List) -> Optional[str]:
        """Encode frame with detection boxes to base64."""
        try:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Draw detection boxes
            if detections:
                draw = ImageDraw.Draw(img)
                enemy_classes = ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']

                for d in detections:
                    class_name = d.get('class_name', getattr(d, 'class_name', 'unknown'))
                    x_center = d.get('x_center', getattr(d, 'x_center', 0.5))
                    y_center = d.get('y_center', getattr(d, 'y_center', 0.5))

                    img_w, img_h = img.size
                    cx = int(x_center * img_w)
                    cy = int(y_center * img_h)
                    box_half = 40

                    if class_name in enemy_classes:
                        color = (255, 0, 0)
                        label = f"ENEMY: {class_name}"
                    elif class_name == 'BonusBox':
                        color = (255, 255, 0)
                        label = "BOX"
                    else:
                        continue

                    draw.rectangle(
                        [cx - box_half, cy - box_half, cx + box_half, cy + box_half],
                        outline=color, width=3
                    )
                    draw.text((cx - box_half, cy - box_half - 15), label, fill=color)

            # Resize for efficiency
            img = img.resize((640, 360), Image.Resampling.LANCZOS)

            # Encode
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"[EnhancedVLM] Encode error: {e}")
            return None

    def get_full_analysis(self, frame: np.ndarray, detections: List,
                          bot_action: Dict, is_attacking: bool,
                          context: Dict = None) -> Dict:
        """
        Run all appropriate analyses based on timing.

        Returns combined results from all levels that ran.

        Args:
            frame: Current game frame
            detections: List of detected objects
            bot_action: Current bot action (for correction saving)
            is_attacking: Whether bot is in attack mode
            context: Additional context for correction saving (mode, health, etc.)
        """
        results = {
            'strategic': None,
            'tactical': None,
            'execution': None,
            'motion': None
        }

        # Build context for correction saving
        ctx = context or {}

        # Strategic (least frequent)
        if self.should_run_strategic():
            results['strategic'] = self.analyze_strategic(
                frame, detections, bot_action, ctx
            )

        # Tactical
        if self.should_run_tactical():
            results['tactical'] = self.analyze_tactical(
                frame, detections, bot_action, ctx
            )

        # Execution (most frequent)
        if self.should_run_execution():
            results['execution'] = self.analyze_execution(
                frame, detections, bot_action, is_attacking, ctx
            )

        # Add frame to buffer
        self.add_frame(frame, detections, {
            'mouse_x': bot_action.get('move_x', bot_action.get('aim_x', 0.5)),
            'mouse_y': bot_action.get('move_y', bot_action.get('aim_y', 0.5)),
            'is_attacking': is_attacking
        })

        return results

    def get_memory_summary(self) -> Dict:
        """Get current memory state for debugging."""
        return {
            'total_critiques': self.memory.total_critiques,
            'good_actions': self.memory.good_actions,
            'bad_actions': self.memory.bad_actions,
            'accuracy': self.memory.good_actions / max(self.memory.total_critiques, 1),
            'current_strategy': self.memory.current_strategy,
            'threat_level': self.memory.threat_level,
            'current_target': self.current_target,
            'current_tactic': self.current_tactic,
            'recent_issues': self.memory.recent_issues[-5:]
        }

    # ═══════════════════════════════════════════════════════════════
    # ASYNC PROCESSING - Don't block the main control loop!
    # ═══════════════════════════════════════════════════════════════

    def start_async(self):
        """Start background analysis thread."""
        self._running = True
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        print("[EnhancedVLM] Background analysis started")

    def stop_async(self):
        """Stop background analysis."""
        self._running = False
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2.0)

    def submit_frame(self, frame: np.ndarray, detections: List,
                     bot_action: Dict, is_attacking: bool, context: Dict = None):
        """
        Submit a frame for async analysis (non-blocking).

        Call this every frame - it will only process when analysis is due.
        Returns immediately, results available via get_last_results().
        """
        with self._lock:
            self._pending_frame = frame.copy() if frame is not None else None
            self._pending_detections = detections
            self._pending_action = bot_action
            self._pending_is_attacking = is_attacking
            self._pending_context = context or {}

        # Also add to frame buffer for motion analysis
        self.add_frame(frame, detections, {
            'mouse_x': bot_action.get('move_x', bot_action.get('aim_x', 0.5)),
            'mouse_y': bot_action.get('move_y', bot_action.get('aim_y', 0.5)),
            'is_attacking': is_attacking
        })

    def get_last_results(self) -> Dict:
        """Get results from last completed analysis (non-blocking)."""
        with self._lock:
            return self._last_results.copy()

    def _analysis_loop(self):
        """Background thread that runs VLM analysis."""
        while self._running:
            try:
                # Get pending frame
                with self._lock:
                    frame = self._pending_frame
                    detections = self._pending_detections
                    action = self._pending_action
                    is_attacking = getattr(self, '_pending_is_attacking', False)
                    context = self._pending_context

                if frame is None:
                    time.sleep(0.1)
                    continue

                # Check what analyses are due
                results = {}
                ctx = context or {}

                # Strategic (least frequent - every 5s)
                if self.should_run_strategic():
                    result = self.analyze_strategic(frame, detections, action, ctx)
                    if result:
                        results['strategic'] = result
                        with self._lock:
                            self._last_results['strategic'] = result

                # Tactical (medium - every 1s)
                if self.should_run_tactical():
                    result = self.analyze_tactical(frame, detections, action, ctx)
                    if result:
                        results['tactical'] = result
                        with self._lock:
                            self._last_results['tactical'] = result

                # Execution (most frequent - every 0.3s)
                if self.should_run_execution():
                    result = self.analyze_execution(frame, detections, action, is_attacking, ctx)
                    if result:
                        results['execution'] = result
                        with self._lock:
                            self._last_results['execution'] = result

                # Small sleep to prevent busy-waiting
                time.sleep(0.05)

            except Exception as e:
                print(f"[EnhancedVLM] Analysis error: {e}")
                time.sleep(0.5)

    # ═══════════════════════════════════════════════════════════════
    # TRAINING DATA GENERATION - Save corrections for model training
    # ═══════════════════════════════════════════════════════════════

    def save_correction(self, bot_action: Dict, vlm_result: Dict,
                        detections: List, context: Dict, level: str = "tactical"):
        """
        Save a VLM correction as training data.

        This is similar to SelfImprover but with richer multi-level analysis.
        Corrections are saved to data/vlm_corrections/ for training.
        """
        if not self._corrections_enabled:
            return

        try:
            # Only save if VLM suggests a different action
            if level == "execution":
                if vlm_result.get('action_correct', True):
                    return  # Action was correct, nothing to learn
                correction = vlm_result.get('correction', {})
            elif level == "tactical":
                # Extract tactical correction
                target = vlm_result.get('priority_target', {})
                tactic = vlm_result.get('recommended_tactic', 'none')
                tactic_params = vlm_result.get('tactic_params', {})

                # Find target position
                target_name = target.get('name', '')
                target_x, target_y = 0.5, 0.5
                for d in detections:
                    d_name = d.get('class_name', getattr(d, 'class_name', ''))
                    if d_name == target_name:
                        target_x = d.get('x_center', getattr(d, 'x_center', 0.5))
                        target_y = d.get('y_center', getattr(d, 'y_center', 0.5))
                        break

                correction = {
                    'move_x': target_x,
                    'move_y': target_y,
                    'should_attack': target.get('type') == 'enemy',
                    'tactic': tactic,
                    'optimal_distance': tactic_params.get('distance', 0.1),
                }
            elif level == "strategic":
                # Strategic corrections affect mode/behavior
                correction = {
                    'recommended_strategy': vlm_result.get('recommended_strategy', 'farm'),
                    'should_flee': vlm_result.get('recommended_strategy') == 'flee',
                    'threat_level': vlm_result.get('threat_assessment', 'unknown'),
                }
            else:
                return

            # Build correction record
            correction_data = {
                'timestamp': time.time(),
                'level': level,
                'bot_action': bot_action,
                'vlm_correction': correction,
                'vlm_full_result': vlm_result,
                # CRITICAL: Include state_vector for training!
                'state_vector': context.get('state_vector'),
                'context': {
                    'mode': context.get('mode', 'PASSIVE'),
                    'health': context.get('health', 1.0),
                    'is_attacking': context.get('is_attacking', False),
                    'num_enemies': len([d for d in detections
                                       if d.get('class_name', getattr(d, 'class_name', ''))
                                       in ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']]),
                    'num_boxes': len([d for d in detections
                                     if d.get('class_name', getattr(d, 'class_name', '')) == 'BonusBox']),
                },
                'memory_context': {
                    'current_strategy': self.memory.current_strategy,
                    'threat_level': self.memory.threat_level,
                    'recent_issues': self.memory.recent_issues[-3:],
                }
            }

            # Save to corrections file
            self._corrections.append(correction_data)
            self._correction_count += 1

            # Periodically save to disk
            if len(self._corrections) >= 10:
                self._flush_corrections()

        except Exception as e:
            print(f"[EnhancedVLM] Error saving correction: {e}")

    def _flush_corrections(self):
        """Write accumulated corrections to disk."""
        if not self._corrections:
            return

        try:
            corrections_dir = Path(__file__).parent.parent / "data" / "vlm_corrections"
            corrections_dir.mkdir(parents=True, exist_ok=True)

            # Find next file number
            existing = list(corrections_dir.glob("enhanced_*.json"))
            next_num = len(existing)

            # Save with timestamp
            filename = f"enhanced_{next_num:04d}_{int(time.time())}.json"
            filepath = corrections_dir / filename

            with open(filepath, 'w') as f:
                json.dump({
                    'source': 'enhanced_vlm',
                    'version': '1.0',
                    'corrections': self._corrections,
                    'session_stats': {
                        'total_critiques': self.memory.total_critiques,
                        'good_actions': self.memory.good_actions,
                        'bad_actions': self.memory.bad_actions,
                    }
                }, f, indent=2)

            print(f"[EnhancedVLM] Saved {len(self._corrections)} corrections to {filename}")
            self._corrections = []

        except Exception as e:
            print(f"[EnhancedVLM] Error flushing corrections: {e}")

    def enable_corrections(self, enabled: bool = True):
        """Enable/disable saving corrections for training."""
        self._corrections_enabled = enabled
        if enabled:
            self._corrections = []
            self._correction_count = 0
            print("[EnhancedVLM] Correction saving ENABLED - will generate training data")

    def stop_and_save(self):
        """Stop and save any remaining corrections."""
        self._flush_corrections()
        print(f"[EnhancedVLM] Session complete: {self._correction_count} total corrections saved")


# Convenience function for quick testing
def test_enhanced_vlm():
    """Test the enhanced VLM system."""
    vlm = EnhancedVLM()

    print("Enhanced VLM initialized")
    print(f"System prompt loaded: {len(vlm._system_prompt)} chars")
    print(f"Strategic interval: {vlm.STRATEGIC_INTERVAL}s")
    print(f"Tactical interval: {vlm.TACTICAL_INTERVAL}s")
    print(f"Execution interval: {vlm.EXECUTION_INTERVAL}s")

    # Test memory
    vlm.memory.add_observation(enemy_type="Sibelon", tactic="kiting")
    vlm.memory.add_observation(issue="too close to enemy")
    print(f"\nMemory context:\n{vlm.memory.get_context_summary()}")


if __name__ == "__main__":
    test_enhanced_vlm()
