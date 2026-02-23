"""
V2 Enhanced VLM - Hierarchical Architecture Aware

VLM analysis specifically designed for V2's hierarchical policy:
- STRATEGIST level: Mode selection feedback (every 5s)
- TACTICIAN level: Target selection with tracked IDs (every 1s)
- EXECUTOR level: Movement quality assessment (every 0.3s)

Key V2-specific features:
1. Uses ByteTrack object IDs for persistent target references
2. Mode-aware prompts aligned with V2 modes (FIGHT/LOOT/FLEE/EXPLORE/CAUTIOUS)
3. Hierarchical feedback matching Strategist→Tactician→Executor flow
4. Richer context from V2's 192-dim state encoding

Usage:
    vlm = VLM_V2(base_url="http://localhost:1234")
    vlm.start_async()

    # In main loop:
    vlm.submit_frame(frame, tracked_objects, policy_output, state_dict)

    # Get recommendations:
    results = vlm.get_last_results()
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

# Import TrackedObject type
try:
    from ..perception.tracker import TrackedObject
except ImportError:
    TrackedObject = None


@dataclass
class V2Memory:
    """V2-specific memory with hierarchical tracking."""
    # Strategist level
    mode_history: List[str] = field(default_factory=list)
    mode_overrides: int = 0  # Times VLM suggested mode change

    # Tactician level
    target_history: List[Dict] = field(default_factory=list)  # {id, class, reason}
    target_switches: int = 0
    missed_targets: List[str] = field(default_factory=list)

    # Executor level
    movement_quality: List[str] = field(default_factory=list)  # good/acceptable/poor
    action_corrections: int = 0

    # Session stats
    total_analyses: int = 0
    strategist_analyses: int = 0
    tactician_analyses: int = 0
    executor_analyses: int = 0

    # Timestamps
    last_strategist: float = 0
    last_tactician: float = 0
    last_executor: float = 0

    def get_context_summary(self) -> str:
        """Get summary for VLM context."""
        lines = []

        if self.mode_history:
            recent_modes = self.mode_history[-5:]
            lines.append(f"Recent modes: {recent_modes}")

        if self.target_history:
            recent_targets = [f"{t['class']}#{t['id']}" for t in self.target_history[-3:]]
            lines.append(f"Recent targets: {recent_targets}")

        if self.missed_targets:
            lines.append(f"Missed opportunities: {self.missed_targets[-3:]}")

        if self.movement_quality:
            quality_counts = {}
            for q in self.movement_quality[-10:]:
                quality_counts[q] = quality_counts.get(q, 0) + 1
            lines.append(f"Movement quality: {quality_counts}")

        lines.append(f"Corrections: {self.action_corrections} | Mode overrides: {self.mode_overrides}")

        return "\n".join(lines) if lines else "(no prior context)"


class VLM_V2:
    """
    V2-specific VLM with hierarchical prompts.

    Provides feedback at three levels matching V2 architecture:
    - Strategist (1Hz): Is the current mode appropriate?
    - Tactician (10Hz): Is the target selection optimal?
    - Executor (60Hz): Is the movement execution correct?
    """

    # Analysis intervals (seconds)
    STRATEGIST_INTERVAL = 5.0   # Match V2 strategist update rate
    TACTICIAN_INTERVAL = 1.0    # Match V2 tactician update rate
    EXECUTOR_INTERVAL = 0.3     # Faster for movement quality

    # V2-specific prompts
    STRATEGIST_PROMPT = """=== V2 STRATEGIST ANALYSIS ===

You are evaluating the bot's MODE SELECTION in a DarkOrbit game.

CURRENT STATE:
- Bot mode: {current_mode}
- Health: {health:.0%}
- Tracked objects: {num_objects} ({num_enemies} enemies, {num_loot} loot)
- Idle time: {idle_time:.1f}s

RECENT COMBAT LOGS (OCR from game):
{combat_logs}

PRIOR CONTEXT:
{memory_context}

V2 MODES:
- FIGHT: Actively attacking enemies
- LOOT: Collecting bonus boxes
- FLEE: Retreating from danger (health < 30% or overwhelming enemies)
- EXPLORE: Moving to find targets (no enemies/loot visible)
- CAUTIOUS: Careful engagement (moderate threat)

Looking at the game screen and combat logs, evaluate:
1. Is {current_mode} the CORRECT mode for this situation?
2. Should the bot switch to a different mode?
3. What threats or opportunities is it missing?

Reply with JSON:
{{
  "current_mode_correct": true,
  "recommended_mode": "FIGHT",
  "mode_reason": "why this mode",
  "threats_detected": ["list of threats"],
  "opportunities_missed": ["missed loot/weak enemies"],
  "urgency": "low/medium/high/critical"
}}"""

    TACTICIAN_PROMPT = """=== V2 TACTICIAN ANALYSIS ===

You are evaluating TARGET SELECTION with tracked object IDs.

CURRENT STATE:
- Mode: {current_mode}
- Current target: {current_target} (ID: {target_id})
- All tracked objects:
{tracked_objects_list}

PRIOR CONTEXT:
{memory_context}

TARGETING RULES:
- In FIGHT mode: Prioritize weakest/closest enemy
- In LOOT mode: Prioritize nearest box
- In FLEE mode: No target, focus on escape direction
- Maintain target until dead/collected (avoid switching)

CRITICAL: You MUST only recommend targets from the "All tracked objects" list above.
DO NOT invent new IDs or recommend objects you see in the screenshot that aren't tracked.
If no good targets exist in the tracked list, set "target_correct": true and keep current target.

Looking at the game screen and tracked objects, evaluate:
1. Is target {current_target} (ID: {target_id}) the BEST choice from the tracked objects?
2. If not, which tracked object ID from the list above should be targeted instead?
3. Is the approach angle/distance appropriate?

Reply with JSON:
{{
  "target_correct": true,
  "recommended_target": {{
    "id": 3,
    "class": "Devo",
    "reason": "closest and weakest FROM TRACKED LIST"
  }},
  "approach": {{
    "distance": "optimal/too_close/too_far",
    "angle": "good/adjust_left/adjust_right",
    "tactic": "orbit/kite/rush/retreat"
  }},
  "priority_order": [3, 5, 7]
}}

NOTE: All IDs in "recommended_target" and "priority_order" MUST be from the tracked objects list above!"""

    EXECUTOR_PROMPT = """=== V2 EXECUTOR ANALYSIS ===

Quick check of MOVEMENT EXECUTION quality.

CONTEXT:
- Mode: {current_mode}
- Target: {current_target} (ID: {target_id})
- Mouse position: ({mouse_x:.2f}, {mouse_y:.2f})
- Is attacking: {is_attacking}
- Last movement: {movement_description}

Was the bot's movement CORRECT for engaging {current_target}?

Reply with JSON:
{{
  "movement_correct": true,
  "quality": "good",
  "issue": "none",
  "correction": {{
    "move_x": 0.5,
    "move_y": 0.5,
    "should_click": true
  }}
}}"""

    def __init__(self,
                 base_url: str = "http://localhost:1234",
                 model: str = "local-model"):
        """
        Initialize V2 VLM.

        Args:
            base_url: LM Studio URL
            model: Model name
        """
        self.base_url = base_url
        self.model = model

        # Memory
        self.memory = V2Memory()

        # Frame buffer for temporal analysis
        self.frame_buffer: deque = deque(maxlen=10)

        # Current state from policy
        self.current_mode = "EXPLORE"
        self.current_target_id: Optional[int] = None
        self.current_target_class: Optional[str] = None

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._analysis_thread: Optional[threading.Thread] = None

        # Pending data
        self._pending_frame = None
        self._pending_tracked_objects: List = []
        self._pending_policy_output: Dict = {}
        self._pending_state_dict: Dict = {}
        self._pending_combat_logs: List[str] = []

        # Results
        self._last_results = {
            'strategist': None,
            'tactician': None,
            'executor': None
        }

        # Correction saving
        self._corrections_enabled = False
        self._corrections = []

        # Load system prompt
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load V2-specific system prompt."""
        prompt_file = Path(__file__).parent.parent.parent / "data" / "vlm_system_prompt_v2.txt"
        try:
            if prompt_file.exists():
                return prompt_file.read_text(encoding='utf-8')
        except:
            pass

        # Default V2 system prompt
        return """You are an expert DarkOrbit game analyst for the V2 hierarchical bot system.

CRITICAL - DO NOT INVENT GAME RULES:
- DMZ has NO PENALTIES for attacking NPCs! DMZ only blocks PvP, not PvE.
- Combat against aliens (Streuner, Lordakia, Mordon, Sibelon, etc.) is ALWAYS allowed everywhere.
- There are NO map zones that restrict attacking NPCs.
- If unsure about a rule, assume combat is allowed.

The V2 bot uses three layers:
1. STRATEGIST: Decides mode (FIGHT/LOOT/FLEE/EXPLORE/CAUTIOUS) based on 60s history
2. TACTICIAN: Selects target from tracked objects using ByteTrack IDs
3. EXECUTOR: Controls precise mouse movement and clicking

You analyze game screenshots to provide feedback at each layer.
Objects are tracked with persistent IDs (e.g., "Devo#50" means Devo enemy with track ID 50).
Be concise and actionable. Focus on what should change, not what's working."""

    def start_async(self):
        """Start background analysis thread."""
        self._running = True
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        print("[VLM-V2] Background analysis started")

    def stop_async(self):
        """Stop background analysis."""
        self._running = False
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2.0)

    def submit_frame(self,
                     frame: np.ndarray,
                     tracked_objects: List,
                     policy_output: Dict,
                     state_dict: Dict,
                     combat_logs: List[str] = None):
        """
        Submit frame for async analysis.

        Args:
            frame: Current game frame (BGR numpy array)
            tracked_objects: List of TrackedObject from ByteTrack
            policy_output: Output from hierarchical policy step()
            state_dict: Output from StateEncoderV2.encode()
            combat_logs: Recent combat log messages from OCR (optional)
        """
        with self._lock:
            self._pending_frame = frame.copy() if frame is not None else None
            self._pending_tracked_objects = tracked_objects
            self._pending_policy_output = policy_output
            self._pending_state_dict = state_dict
            self._pending_combat_logs = combat_logs if combat_logs else []

            # Update current state
            self.current_mode = policy_output.get('mode', 'EXPLORE')
            target_idx = policy_output.get('target_idx', -1)

            if target_idx >= 0 and target_idx < len(tracked_objects):
                target = tracked_objects[target_idx]
                self.current_target_id = getattr(target, 'track_id', target_idx)
                self.current_target_class = getattr(target, 'class_name', 'unknown')
            else:
                self.current_target_id = None
                self.current_target_class = None

    def get_last_results(self) -> Dict:
        """Get results from last completed analyses."""
        with self._lock:
            return self._last_results.copy()

    def should_run_strategist(self) -> bool:
        return time.time() - self.memory.last_strategist > self.STRATEGIST_INTERVAL

    def should_run_tactician(self) -> bool:
        return time.time() - self.memory.last_tactician > self.TACTICIAN_INTERVAL

    def should_run_executor(self) -> bool:
        return time.time() - self.memory.last_executor > self.EXECUTOR_INTERVAL

    def _analysis_loop(self):
        """Background analysis thread."""
        while self._running:
            try:
                # Get pending data
                with self._lock:
                    frame = self._pending_frame
                    tracked_objects = self._pending_tracked_objects
                    policy_output = self._pending_policy_output
                    state_dict = self._pending_state_dict
                    combat_logs = self._pending_combat_logs.copy()

                if frame is None:
                    time.sleep(0.1)
                    continue

                # Run analyses based on timing
                if self.should_run_strategist():
                    result = self._analyze_strategist(frame, tracked_objects, policy_output, state_dict, combat_logs)
                    if result:
                        result['timestamp'] = time.time()  # Add timestamp for dedup
                        with self._lock:
                            self._last_results['strategist'] = result
                        self.memory.strategist_analyses += 1

                if self.should_run_tactician():
                    result = self._analyze_tactician(frame, tracked_objects, policy_output, state_dict)
                    if result:
                        result['timestamp'] = time.time()  # Add timestamp for dedup
                        with self._lock:
                            self._last_results['tactician'] = result
                        self.memory.tactician_analyses += 1

                if self.should_run_executor():
                    result = self._analyze_executor(frame, tracked_objects, policy_output, state_dict)
                    if result:
                        result['timestamp'] = time.time()  # Add timestamp for dedup
                        with self._lock:
                            self._last_results['executor'] = result
                        self.memory.executor_analyses += 1

                time.sleep(0.05)

            except Exception as e:
                print(f"[VLM-V2] Analysis error: {e}")
                time.sleep(0.5)

    def _analyze_strategist(self, frame, tracked_objects, policy_output, state_dict, combat_logs: List[str]) -> Optional[Dict]:
        """Analyze mode selection with combat log context."""
        self.memory.last_strategist = time.time()

        # Count object types
        num_enemies = sum(1 for t in tracked_objects
                        if getattr(t, 'class_name', '') in ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener'])
        num_loot = sum(1 for t in tracked_objects
                      if getattr(t, 'class_name', '') in ['BonusBox', 'box'])

        # Format combat logs for prompt
        combat_logs_str = "\n".join(combat_logs) if combat_logs else "(no recent combat logs)"

        prompt = self.STRATEGIST_PROMPT.format(
            current_mode=self.current_mode,
            health=state_dict.get('player', [1.0])[4] if 'player' in state_dict else 1.0,
            num_objects=len(tracked_objects),
            num_enemies=num_enemies,
            num_loot=num_loot,
            idle_time=0.0,  # TODO: track idle time
            combat_logs=combat_logs_str,
            memory_context=self.memory.get_context_summary()
        )

        result = self._query_vlm(frame, tracked_objects, prompt)

        if result:
            # Update memory
            self.memory.mode_history.append(self.current_mode)
            self.memory.mode_history = self.memory.mode_history[-20:]

            if not result.get('current_mode_correct', True):
                self.memory.mode_overrides += 1
                # Save strategist correction for training
                if self._corrections_enabled:
                    self._save_correction(result, policy_output, state_dict, tracked_objects)

            missed = result.get('opportunities_missed', [])
            if missed:
                self.memory.missed_targets.extend(missed)
                self.memory.missed_targets = self.memory.missed_targets[-10:]

        return result

    def _analyze_tactician(self, frame, tracked_objects, policy_output, state_dict) -> Optional[Dict]:
        """Analyze target selection with tracked IDs."""
        self.memory.last_tactician = time.time()

        # Build tracked objects list for prompt
        objects_list = []
        for i, obj in enumerate(tracked_objects[:10]):  # Limit to 10
            track_id = getattr(obj, 'track_id', i)
            class_name = getattr(obj, 'class_name', 'unknown')
            x = getattr(obj, 'x', 0.5)
            y = getattr(obj, 'y', 0.5)
            conf = getattr(obj, 'confidence', 0.5)
            objects_list.append(f"  - {class_name}#{track_id} at ({x:.2f}, {y:.2f}) conf:{conf:.2f}")

        objects_str = "\n".join(objects_list) if objects_list else "  (no objects tracked)"

        target_str = f"{self.current_target_class}#{self.current_target_id}" if self.current_target_id else "none"

        prompt = self.TACTICIAN_PROMPT.format(
            current_mode=self.current_mode,
            current_target=target_str,
            target_id=self.current_target_id or -1,
            tracked_objects_list=objects_str,
            memory_context=self.memory.get_context_summary()
        )

        result = self._query_vlm(frame, tracked_objects, prompt)

        if result:
            # Update memory
            if self.current_target_id:
                self.memory.target_history.append({
                    'id': self.current_target_id,
                    'class': self.current_target_class,
                    'correct': result.get('target_correct', True)
                })
                self.memory.target_history = self.memory.target_history[-20:]

            if not result.get('target_correct', True):
                self.memory.target_switches += 1
                # Save tactician correction for training
                if self._corrections_enabled:
                    self._save_correction(result, policy_output, state_dict, tracked_objects)

        return result

    def _analyze_executor(self, frame, tracked_objects, policy_output, state_dict) -> Optional[Dict]:
        """Quick movement quality check."""
        self.memory.last_executor = time.time()

        action = policy_output.get('action', {})
        mouse_x = action.get('mouse_x', 0.5)
        mouse_y = action.get('mouse_y', 0.5)
        is_attacking = action.get('should_click', False)

        target_str = f"{self.current_target_class}#{self.current_target_id}" if self.current_target_id else "none"

        prompt = self.EXECUTOR_PROMPT.format(
            current_mode=self.current_mode,
            current_target=target_str,
            target_id=self.current_target_id or -1,
            mouse_x=mouse_x,
            mouse_y=mouse_y,
            is_attacking=is_attacking,
            movement_description="moving toward target"  # TODO: track actual movement
        )

        result = self._query_vlm(frame, tracked_objects, prompt)

        if result:
            quality = result.get('quality', 'acceptable')
            self.memory.movement_quality.append(quality)
            self.memory.movement_quality = self.memory.movement_quality[-50:]

            if not result.get('movement_correct', True):
                self.memory.action_corrections += 1

                # Save correction for training
                if self._corrections_enabled:
                    self._save_correction(result, policy_output, state_dict, tracked_objects)

        return result

    def _query_vlm(self, frame: np.ndarray, tracked_objects: List, prompt: str) -> Optional[Dict]:
        """Query VLM with frame and tracked objects overlay."""
        try:
            # Encode frame with tracked object boxes and IDs
            image_b64 = self._encode_frame_with_tracking(frame, tracked_objects)
            if not image_b64:
                return None

            url = f"{self.base_url}/v1/chat/completions"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ],
                "max_tokens": 400,
                "temperature": 0.2
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

            self.memory.total_analyses += 1
            return json.loads(text)

        except requests.exceptions.Timeout:
            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"[VLM-V2] Query error: {e}")
            return None

    def _encode_frame_with_tracking(self, frame: np.ndarray, tracked_objects: List) -> Optional[str]:
        """Encode frame with tracked object boxes showing IDs."""
        try:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)

            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Draw tracked objects with IDs
            enemy_classes = ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']
            loot_classes = ['BonusBox', 'box']

            img_w, img_h = img.size

            for obj in tracked_objects:
                track_id = getattr(obj, 'track_id', 0)
                class_name = getattr(obj, 'class_name', 'unknown')
                x = getattr(obj, 'x', 0.5)
                y = getattr(obj, 'y', 0.5)
                w = getattr(obj, 'width', 0.05)
                h = getattr(obj, 'height', 0.05)

                # Convert to pixel coordinates
                cx = int(x * img_w)
                cy = int(y * img_h)
                box_w = int(w * img_w / 2)
                box_h = int(h * img_h / 2)

                # Color by class
                if class_name in enemy_classes:
                    color = (255, 0, 0)  # Red for enemies
                    label = f"E:{class_name}#{track_id}"
                elif class_name in loot_classes:
                    color = (255, 255, 0)  # Yellow for loot
                    label = f"L:Box#{track_id}"
                else:
                    continue

                # Draw box
                draw.rectangle(
                    [cx - box_w, cy - box_h, cx + box_w, cy + box_h],
                    outline=color, width=2
                )

                # Draw label with track ID
                draw.text((cx - box_w, cy - box_h - 18), label, fill=color, font=font)

            # Highlight current target
            if self.current_target_id is not None:
                for obj in tracked_objects:
                    if getattr(obj, 'track_id', -1) == self.current_target_id:
                        x = getattr(obj, 'x', 0.5)
                        y = getattr(obj, 'y', 0.5)
                        cx = int(x * img_w)
                        cy = int(y * img_h)
                        # Draw crosshair on current target
                        draw.line([(cx - 30, cy), (cx + 30, cy)], fill=(0, 255, 0), width=3)
                        draw.line([(cx, cy - 30), (cx, cy + 30)], fill=(0, 255, 0), width=3)
                        break

            # Resize for efficiency
            img = img.resize((640, 360), Image.Resampling.LANCZOS)

            # Encode
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"[VLM-V2] Encode error: {e}")
            return None

    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-safe types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            return str(obj)  # Fallback to string representation

    def _save_correction(self, vlm_result: Dict, policy_output: Dict, state_dict: Dict, tracked_objects: List = None):
        """Save correction for V2 training with full object data."""
        # Extract track_ids from tracked_objects for tactician training
        object_track_ids = []
        tracked_objects_snapshot = []  # Full object data snapshot
        if tracked_objects:
            for obj in tracked_objects:
                track_id = getattr(obj, 'track_id', -1)
                object_track_ids.append(track_id)

                # Save full object info for tactician (in case object disappears before training)
                tracked_objects_snapshot.append({
                    'track_id': track_id,
                    'class_name': getattr(obj, 'class_name', 'unknown'),
                    'x': getattr(obj, 'x', 0.5),
                    'y': getattr(obj, 'y', 0.5),
                    'confidence': getattr(obj, 'confidence', 0.0),
                    'feature_vector': getattr(obj, 'feature_vector', [0] * 20)  # Full 20D feature
                })

        correction = {
            'timestamp': time.time(),
            'vlm_result': self._make_json_serializable(vlm_result),
            'policy_output': self._make_json_serializable(policy_output),
            'state_vector': self._make_json_serializable(state_dict.get('full_state', [])),
            # CRITICAL: Save object features for proper executor training
            'objects': self._make_json_serializable(state_dict.get('objects', [])),
            'object_mask': self._make_json_serializable(state_dict.get('object_mask', [])),
            # CRITICAL: Save track_ids for tactician training (to match VLM recommended targets)
            'object_track_ids': object_track_ids,
            # NEW: Save full tracked object snapshot with features
            'tracked_objects_snapshot': self._make_json_serializable(tracked_objects_snapshot),
            'mode': self.current_mode,
            'target_id': self.current_target_id,
            'target_class': self.current_target_class,
            'target_idx': policy_output.get('target_idx', -1)  # Index into objects array
        }

        self._corrections.append(correction)

        # Flush every 10 corrections
        if len(self._corrections) >= 10:
            self._flush_corrections()

    def _flush_corrections(self):
        """Write corrections to disk."""
        if not self._corrections:
            return

        try:
            corrections_dir = Path(__file__).parent.parent.parent / "data" / "vlm_corrections_v2"
            corrections_dir.mkdir(parents=True, exist_ok=True)

            filename = f"v2_corrections_{int(time.time())}.json"
            filepath = corrections_dir / filename

            with open(filepath, 'w') as f:
                json.dump({
                    'source': 'vlm_v2',
                    'corrections': self._corrections,
                    'memory_stats': {
                        'total_analyses': self.memory.total_analyses,
                        'mode_overrides': self.memory.mode_overrides,
                        'target_switches': self.memory.target_switches,
                        'action_corrections': self.memory.action_corrections
                    }
                }, f, indent=2)

            print(f"[VLM-V2] Saved {len(self._corrections)} corrections")
            self._corrections = []

        except Exception as e:
            print(f"[VLM-V2] Save error: {e}")

    def enable_corrections(self, enabled: bool = True):
        """Enable/disable saving corrections."""
        self._corrections_enabled = enabled
        if enabled:
            print("[VLM-V2] Correction saving ENABLED")

    def stop_and_save(self):
        """Stop and save remaining corrections."""
        self._flush_corrections()

    def get_summary(self) -> Dict:
        """Get session summary."""
        return {
            'total_analyses': self.memory.total_analyses,
            'strategist': self.memory.strategist_analyses,
            'tactician': self.memory.tactician_analyses,
            'executor': self.memory.executor_analyses,
            'mode_overrides': self.memory.mode_overrides,
            'target_switches': self.memory.target_switches,
            'action_corrections': self.memory.action_corrections,
            'movement_quality': self._calc_quality_stats()
        }

    def _calc_quality_stats(self) -> Dict:
        """Calculate movement quality statistics."""
        if not self.memory.movement_quality:
            return {'good': 0, 'acceptable': 0, 'poor': 0}

        counts = {'good': 0, 'acceptable': 0, 'poor': 0}
        for q in self.memory.movement_quality:
            if q in counts:
                counts[q] += 1
        return counts


# Test function
def test_vlm_v2():
    """Test V2 VLM initialization."""
    vlm = VLM_V2()
    print("V2 VLM initialized")
    print(f"System prompt: {len(vlm._system_prompt)} chars")
    print(f"Strategist interval: {vlm.STRATEGIST_INTERVAL}s")
    print(f"Tactician interval: {vlm.TACTICIAN_INTERVAL}s")
    print(f"Executor interval: {vlm.EXECUTOR_INTERVAL}s")


if __name__ == "__main__":
    test_vlm_v2()
