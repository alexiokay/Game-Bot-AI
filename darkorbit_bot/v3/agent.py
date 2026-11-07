"""
V3 agent: record your play, or let the single policy play.

    python -m darkorbit_bot.v3.agent --mode record                 # you play, it records
    python -m darkorbit_bot.v3.agent --mode bot                    # policy plays
    python -m darkorbit_bot.v3.agent --mode bot --dry-run          # decides, sends nothing
    python -m darkorbit_bot.v3.agent --game my_game --mode record  # any game with a profile

Hotkeys:  F1 pause   F2 takeover (bot mode: you play, recorded as a correction)   F3 stop

Every run records a session (video + frames.jsonl + meta) under
darkorbit_bot/data/v3_sessions/<game>/. Game-specific settings (detector,
class roles, HUD bars, keys, button behaviour) come from games/<game>/profile.yaml. The control loop runs on the main thread;
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
from .profile import GameProfile, load_profile, DEFAULT_GAME
from .schema import HUMAN, BOT, TAKEOVER, sessions_dir, default_policy_path
from .session_io import SessionWriter, file_sha1

PAUSED = "paused"   # frames not learned from


def _rmb_down() -> bool:
    try:
        return ctypes.windll.user32.GetAsyncKeyState(0x02) < 0
    except Exception:
        return False


def hud_reader_for(profile: GameProfile):
    """HUDReader (colour-based) configured from the profile's bars, or None."""
    if profile.hp_bar is None and profile.shield_bar is None:
        return None
    from ..v2.perception.hud_ocr import HUDReader, HUDConfig
    cfg = HUDConfig(ocr_backend="color_only")
    none_region = (0.0, 0.0, 0.0, 0.0)   # empty region = bar not read (unknown)
    hp, sh = profile.hp_bar, profile.shield_bar
    cfg.hp_bar_region = hp.region if hp else none_region
    if hp:
        cfg.hp_color_low, cfg.hp_color_high = hp.hsv_low, hp.hsv_high
        cfg.hp_color_critical_low, cfg.hp_color_critical_high = hp.alt_hsv_low, hp.alt_hsv_high
    cfg.shield_bar_region = sh.region if sh else none_region
    if sh:
        cfg.shield_color_low, cfg.shield_color_high = sh.hsv_low, sh.hsv_high
    return HUDReader(cfg)


def check_policy_for_profile(model, profile: GameProfile):
    """Refuse a policy trained for another game or key list."""
    cfg = model.cfg
    if cfg.game and cfg.game != profile.name:
        raise SystemExit(f"This policy was trained for '{cfg.game}', not '{profile.name}'.")
    if tuple(cfg.keys) != tuple(profile.keys):
        raise SystemExit("The policy's key list differs from the profile's controls.keys - retrain it.")


class Agent:
    def __init__(self, mode: str, capture, detector, writer: SessionWriter,
                 input_capture, profile: GameProfile, hud_reader=None, runner=None,
                 mouse=None, keyboard=None, movement=None,
                 monitor: Optional[Dict] = None, fps: Optional[int] = None,
                 dry_run: bool = False, record_frames: bool = True, broadcaster=None):
        assert mode in ('record', 'bot')
        self.mode = mode
        self.profile = profile
        self.broadcaster = broadcaster   # overlay.StateBroadcaster for the live viewer
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
        self.fps = fps or profile.fps
        self.dry_run = dry_run
        self.record_frames = record_frames

        self.features = FeatureBuilder(profile)
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
            # Buttons: 'hold' = the predicted state is held down (DarkOrbit: hold
            # left to fly toward the cursor); 'click' = a short tap when the
            # prediction turns on
            modes = {'left': self.profile.left_button, 'right': self.profile.right_button}
            for want, attr, button in ((action['lmb'], '_lmb_down', 'left'), (action['rmb'], '_rmb_down', 'right')):
                was = getattr(self, attr)
                if modes[button] == 'hold':
                    if want != was:
                        self.mouse._send_click(want, button)
                elif want and not was:
                    self.mouse.click(button=button, hold_duration=0.05)
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

        if controller in (BOT, TAKEOVER) and self.runner is not None:
            action = self.runner.act(feats)
            decision = {k: action[k] for k in ('mx', 'my', 'lmb', 'rmb', 'keys')}
            record['bot_target'] = [action['target_track_id'], round(action['target_prob'], 3)]
            if controller == BOT:
                record['bot_action'] = decision
                self._execute(action)
            else:
                # Takeover: you play; keep what the bot WOULD have done, for the
                # overlay and for spotting where it disagrees with you
                record['bot_suggestion'] = decision

        self.writer.add(record, frame if self.record_frames else None)
        if self.broadcaster is not None and self.broadcaster.has_clients():
            from .overlay import build_state
            self.broadcaster.publish(
                build_state(record, feats, self.profile,
                            self.runner.last_debug if self.runner is not None else None),
                frame)
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
        if self.broadcaster is not None:
            try:
                self.broadcaster.close()
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
    ap.add_argument('--game', default=DEFAULT_GAME, help='game profile name (games/<name>/profile.yaml) or path')
    ap.add_argument('--mode', choices=['record', 'bot'], default='record')
    ap.add_argument('--model', default=None, help='YOLO weights (default: from the profile)')
    ap.add_argument('--policy', default=None, help='V3 policy (default: models/v3/<game>/policy.pt)')
    ap.add_argument('--monitor', type=int, default=1)
    ap.add_argument('--fps', type=int, default=None, help='default: from the profile')
    ap.add_argument('--sessions-dir', default=None, help='default: darkorbit_bot/data/v3_sessions/<game>')
    ap.add_argument('--no-video', action='store_true', help='record detections/inputs only')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='bot decides but sends no input')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--viewer-port', type=int, default=9998, help='live overlay port (127.0.0.1 only)')
    ap.add_argument('--no-broadcast', action='store_true', help='disable the live overlay server')
    args = ap.parse_args(argv)

    profile = load_profile(args.game)
    fps = args.fps or profile.fps
    model_path = args.model or profile.detector_path()
    policy_path = Path(args.policy) if args.policy else default_policy_path(profile.name)

    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    # Perception + input capture (reused from V1/V2)
    from ..detection.detector import ScreenCapture, GameDetector
    from ..v2.training.shadow_trainer import HumanInputCapture

    capture = ScreenCapture(monitor_index=args.monitor)
    mon = capture.monitor
    detector = GameDetector(model_path, confidence_threshold=profile.detector_confidence)
    hud = None if args.no_hud else hud_reader_for(profile)
    inp = HumanInputCapture(mon['left'], mon['top'], mon['width'], mon['height'], confine_mouse=False)

    runner = mouse = keyboard = movement = None
    if args.mode == 'bot':
        from .model import load_policy, PolicyRunner
        if not policy_path.exists():
            raise SystemExit(f"No V3 policy at {policy_path}. Record sessions and train first.")
        model = load_policy(policy_path, device)
        check_policy_for_profile(model, profile)
        runner = PolicyRunner(model, device=device)
        if not args.dry_run:
            from ..movement.generator import (MovementGenerator, MouseController,
                                              MovementProfile, KeyboardController)
            prof_path = Path(__file__).resolve().parents[1] / "data" / "my_movement_profile.json"
            try:
                mprof = MovementProfile.load(str(prof_path)) if prof_path.exists() else MovementProfile()
            except Exception:
                mprof = MovementProfile()
            movement = MovementGenerator(mprof)
            mouse = MouseController(mon['width'], mon['height'], game_rect=mon)
            keyboard = KeyboardController()

    root = Path(args.sessions_dir) if args.sessions_dir else sessions_dir(profile.name)
    writer = SessionWriter({
        'mode': args.mode, 'dry_run': args.dry_run, 'monitor': mon,
        'yolo_model': str(model_path), 'yolo_sha1': file_sha1(model_path),
        'class_names': getattr(detector, 'class_names', None),
        'policy': str(policy_path) if args.mode == 'bot' else None,
    }, root=root, record_video=not args.no_video, fps=fps, video_size=profile.video_size, profile=profile)

    broadcaster = None
    if not args.no_broadcast:
        from .overlay import StateBroadcaster
        try:
            broadcaster = StateBroadcaster(port=args.viewer_port)
            print(f'[V3] Live overlay on 127.0.0.1:{args.viewer_port} (python -m darkorbit_bot.v3.viewer --live)')
        except OSError as e:
            print(f'[V3] Live overlay disabled: {e}')

    agent = Agent(args.mode, capture, detector, writer, inp, profile, hud_reader=hud, runner=runner,
                  mouse=mouse, keyboard=keyboard, movement=movement, monitor=mon,
                  fps=fps, dry_run=args.dry_run, record_frames=not args.no_video,
                  broadcaster=broadcaster)

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

    print(f"[V3] Game: {profile.name} | mode: {args.mode}{' (dry run)' if args.dry_run else ''} "
          f"| session {writer.session_id}")
    print("[V3] F1 pause | F2 takeover (bot mode) | F3 stop")
    agent.start_hotkeys()
    agent.run()


if __name__ == '__main__':
    main()
