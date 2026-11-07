"""
V3 overlay viewer - see what the policy sees and decides.

    # live, next to a running agent (record or bot mode)
    python -m darkorbit_bot.v3.viewer --live

    # replay a recorded session; with --policy, shows what that policy WOULD
    # have done at every moment and prints how often it agrees with you
    python -m darkorbit_bot.v3.viewer --replay darkorbit_bot/data/v3_sessions/darkorbit/<id> --policy models/v3/darkorbit/policy.pt

    # render the replay to a video file instead of a window
    python -m darkorbit_bot.v3.viewer --replay <session> --policy <policy> --export overlay.mp4

Window keys:  q quit   space pause   n next frame (paused)   p toggle UI/self boxes   + / - speed
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .overlay import build_state, draw_overlay, StateClient

WINDOW = "V3 overlay"


def _show(img, delay_ms: int) -> int:
    import cv2
    cv2.imshow(WINDOW, img)
    return cv2.waitKey(max(1, delay_ms)) & 0xFF


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------

def run_live(port: int, size=(960, 540)):
    import cv2
    client = StateClient(port=port)
    show_passive = True
    print(f"[VIEWER] Connecting to 127.0.0.1:{port} (start the agent; q to quit)")
    while True:
        try:
            client.connect()
            print("[VIEWER] Connected")
            while True:
                state, frame = client.receive()
                k = _show(draw_overlay(frame, state, size, show_passive), 1)
                if k == ord('q'):
                    return
                if k == ord('p'):
                    show_passive = not show_passive
        except (ConnectionError, OSError):
            client.close()
            blank = np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(blank, "waiting for agent...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if _show(blank, 1000) == ord('q'):
                return
        finally:
            pass


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

class _VideoReader:
    """Sequential reader that returns the frame for an increasing frame_idx."""

    def __init__(self, path: Path):
        import cv2
        self.cap = cv2.VideoCapture(str(path)) if path.exists() else None
        self.pos = -1
        self.last = None

    def get(self, idx: Optional[int]):
        if idx is None or self.cap is None:
            return self.last            # dropped frame: keep showing the previous one
        while self.pos < idx:
            ok, f = self.cap.read()
            if not ok:
                return self.last
            self.pos += 1
            self.last = f
        return self.last

    def close(self):
        if self.cap is not None:
            self.cap.release()


def replay(session_dir, policy_path: Optional[str] = None, game: Optional[str] = None,
           export: Optional[str] = None, max_frames: Optional[int] = None,
           speed: float = 1.0, show_window: bool = True) -> Dict:
    """Replays a session with the overlay. Returns agreement stats (with a policy)."""
    import cv2
    from .features import FeatureBuilder
    from .profile import load_profile
    from .session_io import SessionReader

    reader = SessionReader(session_dir)
    profile = load_profile(game or reader.game or 'darkorbit')
    fb = FeatureBuilder(profile)
    runner = None
    if policy_path:
        from .agent import check_policy_for_profile
        from .model import load_policy, PolicyRunner
        model = load_policy(policy_path)
        check_policy_for_profile(model, profile)
        runner = PolicyRunner(model)

    size = tuple(reader.meta.get('video_size') or profile.video_size)
    video = _VideoReader(reader.video_path)
    writer = None
    if export:
        Path(export).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(export), cv2.VideoWriter_fourcc(*'mp4v'), reader.meta.get('fps', 20), size)

    fps = float(reader.meta.get('fps', 20))
    paused, show_passive = False, True
    aim_err, click_agree, n_cmp, frames = [], 0, 0, 0
    try:
        for rec in reader.records():
            feats = fb.step(rec)
            debug = None
            if runner is not None and rec.get('controller') != 'paused':
                a = runner.act(feats)
                debug = runner.last_debug
                if rec.get('controller') in ('human', 'takeover') and rec.get('input'):
                    # Show the policy's decision as a suggestion next to your input
                    rec = dict(rec)
                    rec['bot_suggestion'] = {k: a[k] for k in ('mx', 'my', 'lmb', 'rmb', 'keys')}
                    rec.pop('bot_action', None)
                    inp = rec['input']
                    aim_err.append(math.hypot(a['mx'] - inp['mx'], a['my'] - inp['my']))
                    click_agree += int(bool(a['lmb']) == bool(inp.get('lmb')))
                    n_cmp += 1
            img = draw_overlay(video.get(rec.get('frame_idx')), build_state(rec, feats, profile, debug),
                               size, show_passive)
            frames += 1
            if writer is not None:
                writer.write(img)
            if show_window and writer is None:
                delay = int(1000 / (fps * speed))
                while True:
                    k = _show(img, 0 if paused else delay)
                    if k == ord('q'):
                        return _stats(aim_err, click_agree, n_cmp, frames)
                    if k == ord(' '):
                        paused = not paused
                    elif k == ord('p'):
                        show_passive = not show_passive
                    elif k in (ord('+'), ord('=')):
                        speed = min(8.0, speed * 2)
                    elif k == ord('-'):
                        speed = max(0.125, speed / 2)
                    if not paused or k == ord('n'):
                        break
            if max_frames is not None and frames >= max_frames:
                break
    finally:
        video.close()
        if writer is not None:
            writer.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return _stats(aim_err, click_agree, n_cmp, frames)


def _stats(aim_err, click_agree, n_cmp, frames) -> Dict:
    s = {'frames': frames, 'compared_frames': n_cmp}
    if n_cmp:
        s['mean_aim_error'] = float(np.mean(aim_err))       # screen fraction
        s['aim_within_5pct'] = float(np.mean(np.array(aim_err) < 0.05))
        s['left_button_agreement'] = click_agree / n_cmp
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3 overlay viewer (live or replay)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--live', action='store_true', help='connect to a running agent')
    g.add_argument('--replay', help='session folder to replay')
    ap.add_argument('--port', type=int, default=9998)
    ap.add_argument('--policy', default=None, help='replay: show what this policy would do')
    ap.add_argument('--game', default=None, help="replay: profile to use (default: the session's game)")
    ap.add_argument('--export', default=None, help='replay: write the overlay to this .mp4 instead of a window')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--speed', type=float, default=1.0)
    args = ap.parse_args(argv)

    if args.live:
        run_live(args.port)
        return
    stats = replay(args.replay, args.policy, args.game, args.export, args.max_frames, args.speed,
                   show_window=args.export is None)
    print(f"[VIEWER] Replayed {stats['frames']} frames" + (f" -> {args.export}" if args.export else ""))
    if stats.get('compared_frames'):
        print(f"[VIEWER] Policy vs you on {stats['compared_frames']} frames: "
              f"mean aim error {stats['mean_aim_error']:.3f} of the screen, "
              f"within 5%: {stats['aim_within_5pct'] * 100:.0f}%, "
              f"left button agrees: {stats['left_button_agreement'] * 100:.0f}%")


if __name__ == '__main__':
    main()
