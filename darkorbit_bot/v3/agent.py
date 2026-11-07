"""
V3 agent: record your play, or let the single policy play.

    python -m darkorbit_bot.v3.agent --mode record            # you play, it records
    python -m darkorbit_bot.v3.agent --mode bot               # policy plays
    python -m darkorbit_bot.v3.agent --mode bot --dry-run     # policy decides, nothing is sent

Hotkeys:  F1 pause   F2 takeover (bot mode: you play, recorded as a correction)   F3 stop

Every run records a session (video + frames.jsonl + meta) under
darkorbit_bot/data/v3_sessions/. The control loop runs on the main thread;
hotkeys and signals only set flags, so stop() always finishes (inputs
released, session closed) before the process exits.
"""

import argparse
import ctypes
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .features import FeatureBuilder, detection_to_raw
from .schema import (RECORD_FPS, SESSIONS_DIR, DEFAULT_POLICY_PATH, DEFAULT_YOLO_MODEL,
                     HUMAN, BOT, TAKEOVER)
from .session_io import SessionWriter, file_sha1

PAUSED = "paused"   # frames not learned from


def _rmb_down() -> bool:
    try:
        return ctypes.windll.user32.GetAsyncKeyState(0x02) < 0
    except Exception:
        return False


class Agent:
    def __init__(self, mode: str, capture, detector, writer: SessionWriter,
                 input_capture, hud_reader=None, runner=None,
                 mouse=None, keyboard=None, movement=None,
                 monitor: Optional[Dict] = None, fps: int = RECORD_FPS,
                 dry_run: bool = False, record_frames: bool = True):
        assert mode in ('record', 'bot')
        self.mode = mode
        self.capture = capture
        self.detector = detector
        self.writer = writer
        self.input_capture = input_capture
        self.hud_reader = hud_reader
        self.runner = runner
        self.mouse = mouse
        self.keyboard = keyboard
        self.movement = movement
        self.monitor = monitor or {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
        self.fps = fps
        self.dry_run = dry_run
        self.record_frames = record_frames

        self.features = FeatureBuilder()
        self.running = False
        self.paused = False
        self.takeover = False
        self._stopped = False
        self._lmb_down = False
        self._rmb_down = False
        self._held_keys = set()
        self._t0 = None
        self.frames = 0
        self.errors = 0
        self._listener = None

    # ---------------------------------------------------------------- hotkeys
    def toggle_pause(self):
        self.paused = not self.paused
        print(f"\n[V3] {'PAUSED' if self.paused else 'RESUMED'}")
        self._release_inputs()

    def toggle_takeover(self):
        if self.mode != 'bot':
            return
        self.takeover = not self.takeover
        self._release_inputs()
        if self.runner is not None:
            self.runner.reset()
        print(f"\n[V3] {'TAKEOVER - you are playing (recorded as correction)' if self.takeover else 'BOT in control'}")

    def request_stop(self):
        print("\n[V3] Stop requested")
        self.running = False

    def start_hotkeys(self):
        try:
            from pynput import keyboard as kb
        except ImportError:
            print("[V3] pynput not available - hotkeys disabled (Ctrl+C to stop)")
            return
        actions = {kb.Key.f1: self.toggle_pause, kb.Key.f2: self.toggle_takeover, kb.Key.f3: self.request_stop}

        def on_press(key):
            fn = actions.get(key)
            if fn:
                fn()
        self._listener = kb.Listener(on_press=on_press, daemon=True)
        self._listener.start()

    # ------------------------------------------------------------- inputs out
    def _release_inputs(self):
        """Release everything the bot may be holding (safe to call any time)."""
        self._lmb_down = self._rmb_down = False
        self._held_keys.clear()
        if self.dry_run:
            return
        try:
            if self.keyboard is not None:
                self.keyboard.release_all()
        except Exception as e:
            print(f"[V3] could not release keys: {e}")
        try:
            if self.mouse is not None:
                self.mouse.release_buttons()
        except Exception as e:
            print(f"[V3] could not release mouse buttons: {e}")

    def _current_cursor(self):
        x, y = self.input_capture.get_mouse_position()
        return x, y

    def _execute(self, action: Dict):
        if self.dry_run:
            return
        # Mouse movement (human-like path, non-blocking)
        if self.mouse is not None:
            cx, cy = self._current_cursor()
            tx, ty = float(np.clip(action['mx'], 0, 1)), float(np.clip(action['my'], 0, 1))
            dist = float(np.hypot(tx - cx, ty - cy))
            if dist > 0.01:
                if self.movement is not None:
                    duration = max(0.05, min(0.2, dist * 0.5))
                    path = self.movement.generate_path((cx, cy), (tx, ty), target_duration=duration)
                    self.mouse.execute_path_smooth(path)
                else:
                    self.mouse.move_to(tx, ty)
            # Buttons as held state (DarkOrbit: hold left = fly toward cursor)
            for want, attr, button in ((action['lmb'], '_lmb_down', 'left'), (action['rmb'], '_rmb_down', 'right')):
                if want != getattr(self, attr):
                    self.mouse._send_click(want, button)
                    setattr(self, attr, want)
        # Keys as held state
        if self.keyboard is not None:
            want = set(action['keys'])
            for k in want - self._held_keys:
                self.keyboard.hold_key(k)
            for k in self._held_keys - want:
                self.keyboard.release_key(k)
            self._held_keys = want

    # ------------------------------------------------------------------- loop
    def _read_input(self) -> Dict:
        a = self.input_capture.get_action()
        return {
            'mx': round(float(a['mouse_x']), 5), 'my': round(float(a['mouse_y']), 5),
            'lmb': bool(a['should_click']), 'rmb': _rmb_down(),
            'keys': sorted(k for k, v in a.get('keyboard', {}).items() if v),
        }

    def step(self):
        frame = self.capture.capture()
        if frame is None:
            return
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        dets = self.detector.detect_frame(frame)

        hud = None
        if self.hud_reader is not None:
            v = self.hud_reader.read(frame)
            hud = {'hp': round(float(v.hp), 4), 'shield': round(float(v.shield), 4),
                   'hp_conf': round(float(v.hp_confidence), 3),
                   'shield_conf': round(float(v.shield_confidence), 3)}

        if self.paused:
            controller = PAUSED
        elif self.mode == 'record':
            controller = HUMAN
        else:
            controller = TAKEOVER if self.takeover else BOT

        record = {
            't': round(now - self._t0, 4), 'wall': round(time.time(), 3),
            'controller': controller,
            'dets': [detection_to_raw(d) for d in dets],
            'hud': hud,
            'input': self._read_input(),
        }
        feats = self.features.step(record)

        if controller == BOT and self.runner is not None:
            action = self.runner.act(feats)
            record['bot_action'] = {k: action[k] for k in ('mx', 'my', 'lmb', 'rmb', 'keys')}
            record['bot_target'] = [action['target_idx'], round(action['target_prob'], 3)]
            self._execute(action)

        self.writer.add(record, frame if self.record_frames else None)
        self.frames += 1
        if self.frames % (self.fps * 5) == 0:
            n = int(feats['mask'].sum())
            print(f"[V3] {self.frames} frames | {controller} | objects {n} | "
                  f"video {self.writer.n_video_frames} (dropped {self.writer.n_dropped})")

    def run(self, max_frames: Optional[int] = None):
        self.running = True
        period = 1.0 / self.fps
        consecutive_errors = 0
        try:
            while self.running:
                t0 = time.perf_counter()
                try:
                    self.step()
                    consecutive_errors = 0
                except Exception as e:
                    self.errors += 1
                    consecutive_errors += 1
                    self._release_inputs()
                    print(f"[V3] frame error: {e}")
                    if consecutive_errors >= 50:
                        print("[V3] 50 errors in a row - stopping")
                        break
                if max_frames is not None and self.frames >= max_frames:
                    break
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Release inputs, close the session (runs once)."""
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self._release_inputs()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        try:
            self.writer.close()
            print(f"[V3] Session saved: {self.writer.dir} ({self.writer.n_records} frames, "
                  f"{self.writer.n_video_frames} video frames)")
        except Exception as e:
            print(f"[V3] could not close session: {e}")
        try:
            if hasattr(self.capture, 'stop'):
                self.capture.stop()
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3 agent: record or play")
    ap.add_argument('--mode', choices=['record', 'bot'], default='record')
    ap.add_argument('--model', default=DEFAULT_YOLO_MODEL, help='YOLO weights')
    ap.add_argument('--policy', default=str(DEFAULT_POLICY_PATH), help='V3 policy (bot mode)')
    ap.add_argument('--monitor', type=int, default=1)
    ap.add_argument('--fps', type=int, default=RECORD_FPS)
    ap.add_argument('--sessions-dir', default=str(SESSIONS_DIR))
    ap.add_argument('--no-video', action='store_true', help='record detections/inputs only')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='bot decides but sends no input')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args(argv)

    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    # Perception + input capture (reused from V1/V2)
    from ..detection.detector import ScreenCapture, GameDetector
    from ..v2.perception.hud_ocr import create_hud_reader
    from ..v2.training.shadow_trainer import HumanInputCapture

    capture = ScreenCapture(monitor_index=args.monitor)
    mon = capture.monitor
    detector = GameDetector(args.model)
    hud = None if args.no_hud else create_hud_reader(ocr_backend="color_only")
    inp = HumanInputCapture(mon['left'], mon['top'], mon['width'], mon['height'], confine_mouse=False)

    runner = mouse = keyboard = movement = None
    if args.mode == 'bot':
        from .model import load_policy, PolicyRunner
        if not Path(args.policy).exists():
            raise SystemExit(f"No V3 policy at {args.policy}. Record sessions and train first.")
        runner = PolicyRunner(load_policy(args.policy, device), device=device)
        if not args.dry_run:
            from ..movement.generator import (MovementGenerator, MouseController,
                                              MovementProfile, KeyboardController)
            prof_path = Path(__file__).resolve().parents[1] / "data" / "my_movement_profile.json"
            try:
                profile = MovementProfile.load(str(prof_path)) if prof_path.exists() else MovementProfile()
            except Exception:
                profile = MovementProfile()
            movement = MovementGenerator(profile)
            mouse = MouseController(mon['width'], mon['height'], game_rect=mon)
            keyboard = KeyboardController()

    writer = SessionWriter({
        'mode': args.mode, 'dry_run': args.dry_run, 'monitor': mon,
        'yolo_model': str(args.model), 'yolo_sha1': file_sha1(args.model),
        'class_names': getattr(detector, 'class_names', None),
        'policy': args.policy if args.mode == 'bot' else None,
    }, root=Path(args.sessions_dir), record_video=not args.no_video, fps=args.fps)

    agent = Agent(args.mode, capture, detector, writer, inp, hud_reader=hud, runner=runner,
                  mouse=mouse, keyboard=keyboard, movement=movement, monitor=mon,
                  fps=args.fps, dry_run=args.dry_run, record_frames=not args.no_video)

    # Ctrl+Break (launcher Stop) / SIGTERM: only ask the loop to stop
    def _sig(signum, frame):
        agent.request_stop()
    for name in ('SIGBREAK', 'SIGTERM'):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _sig)
            except (ValueError, OSError):
                pass

    print(f"[V3] Mode: {args.mode}{' (dry run)' if args.dry_run else ''} | session {writer.session_id}")
    print("[V3] F1 pause | F2 takeover (bot mode) | F3 stop")
    agent.start_hotkeys()
    agent.run()


if __name__ == '__main__':
    main()
