"""
The ONE place V3 model inputs are built - for any game, via its GameProfile.

Both the live agent and the training dataset feed per-frame records (the same
dicts written to frames.jsonl) through FeatureBuilder.step(), in time order.
That guarantees training inputs and runtime inputs are identical - including
the tracker, the object order and the previous-action token.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..v2.perception.tracker import ObjectTracker
from .profile import GameProfile, TOKEN_ROLES, class_hash
from .schema import (MAX_OBJECTS, OBJECT_DIM, OBJECT_BASE_DIM, PLAYER_DIM, OBJ_VELOCITY_SCALE,
                     TRACK_MATCH_IOU, TRACK_MAX_MISSED)

assert OBJECT_DIM == OBJECT_BASE_DIM + len(TOKEN_ROLES)
ROLE_INDEX = {r: i for i, r in enumerate(TOKEN_ROLES)}


# --------------------------------------------------------------------------
# Raw record helpers (shared by agent and dataset)
# --------------------------------------------------------------------------

def detection_to_raw(det) -> Dict:
    """YOLO Detection (detection/detector.py) -> compact JSON-able dict.
    Stores the raw class name; roles are applied later from the profile, so a
    profile edit never requires re-recording."""
    return {
        'c': det.class_name,
        'p': round(float(det.confidence), 4),
        'b': [round(float(det.x_min), 5), round(float(det.y_min), 5),
              round(float(det.x_max), 5), round(float(det.y_max), 5)],
    }


def raw_to_tracker_input(raw: Dict) -> Dict:
    """Compact stored detection -> dict ObjectTracker.update() understands."""
    x1, y1, x2, y2 = raw['b']
    return {
        'x_center': (x1 + x2) / 2, 'y_center': (y1 + y2) / 2,
        'width': x2 - x1, 'height': y2 - y1,
        'confidence': raw['p'], 'class_name': raw['c'],
    }


def player_position(raw_dets: List[Dict], profile: GameProfile):
    """Our character's position from the detector (role 'self'). When not seen:
    screen centre (profile self_fallback 'center') or (0.5, 0.5) marked unseen."""
    best = None
    for d in raw_dets:
        if profile.role_of(d['c']) == 'self' and (best is None or d['p'] > best['p']):
            best = d
    if best is None:
        return 0.5, 0.5, False
    x1, y1, x2, y2 = best['b']
    return (x1 + x2) / 2, (y1 + y2) / 2, True


def action_vector(inp: Optional[Dict], keys: Sequence[str]) -> np.ndarray:
    """Input dict {mx, my, lmb, rmb, keys:[...]} -> [4 + len(keys)] float vector
    (mouse x, mouse y, left, right, one value per profile key)."""
    v = np.zeros(4 + len(keys), dtype=np.float32)
    if not inp:
        v[0] = v[1] = 0.5
        return v
    v[0] = float(np.clip(inp.get('mx', 0.5), 0.0, 1.0))
    v[1] = float(np.clip(inp.get('my', 0.5), 0.0, 1.0))
    v[2] = 1.0 if inp.get('lmb') else 0.0
    v[3] = 1.0 if inp.get('rmb') else 0.0
    pressed = set(inp.get('keys', ()))
    for i, k in enumerate(keys):
        if k in pressed:
            v[4 + i] = 1.0
    return v


def acted_input(record: Optional[Dict]) -> Optional[Dict]:
    """The input that actually happened this frame: the bot's action on bot
    frames, the human's input otherwise."""
    if not record:
        return None
    if record.get('controller') == 'bot' and record.get('bot_action'):
        return record['bot_action']
    return record.get('input')


def object_features(o, role: str, px: float, py: float) -> np.ndarray:
    """[OBJECT_DIM] token for one tracked object, relative to our character.
    [0-3] x, y, distance, angle   [4-7] vx, vy, speed, heading
    [8-11] w, h, confidence, age  [12-15] hits, time since update, tracked, lost
    [16-22] role one-hot (TOKEN_ROLES)"""
    dx, dy = o.x - px, o.y - py
    f = np.zeros(OBJECT_DIM, dtype=np.float32)
    f[0:16] = (
        o.x, o.y, np.hypot(dx, dy), np.arctan2(dy, dx) / np.pi,
        o.vx * OBJ_VELOCITY_SCALE, o.vy * OBJ_VELOCITY_SCALE,
        o.get_speed() * OBJ_VELOCITY_SCALE, o.get_heading() / np.pi,
        o.width, o.height, o.confidence, min(o.age / 100, 1.0),
        min(o.hits / 50, 1.0), min(o.time_since_update / 10, 1.0),
        1.0 if o.state == "tracked" else 0.0, 1.0 if o.state == "lost" else 0.0,
    )
    f[OBJECT_BASE_DIM + ROLE_INDEX.get(role, ROLE_INDEX['other'])] = 1.0
    return f


# --------------------------------------------------------------------------
# Feature builder
# --------------------------------------------------------------------------

class FeatureBuilder:
    """Stateful per-session feature builder. Call step() once per frame, in order."""

    def __init__(self, profile: GameProfile):
        self.profile = profile
        self.keys = tuple(profile.keys)
        self.tracker = ObjectTracker(match_thresh=TRACK_MATCH_IOU)
        self.reset()

    def reset(self):
        self.tracker.reset()
        self._prev_t = None
        self._prev_xy = None
        self._prev_hp = None
        self._prev_shield = None
        self._last_damage_t = None
        self._prev_record = None

    def step(self, record: Dict) -> Dict:
        """
        record: one frames.jsonl line (dict) with at least
            t, dets, hud {hp, shield, hp_conf}, controller, input / bot_action
        Returns dict with numpy arrays:
            player [PLAYER_DIM], objects [MAX_OBJECTS, OBJECT_DIM],
            classes [MAX_OBJECTS] (class-name hash), mask [MAX_OBJECTS],
            prev_action [4 + K], tracks (TrackedObject list in token order), player_xy
        """
        prof = self.profile
        t = float(record.get('t', 0.0))
        raw_dets = record.get('dets', [])
        px, py, seen = player_position(raw_dets, prof)

        # Track every object that is neither us nor ignored; the tracker is
        # deterministic, so replaying stored detections reproduces live tracks
        tracker_in = [raw_to_tracker_input(d) for d in raw_dets
                      if prof.role_of(d['c']) not in ('self', 'ignore')]
        tracks = self.tracker.update(tracker_in)
        # Only objects seen recently (no long-lost predicted ghosts)
        tracks = [o for o in tracks if o.time_since_update <= TRACK_MAX_MISSED]
        tracks = sorted(tracks, key=lambda o: (o.x - px) ** 2 + (o.y - py) ** 2)[:MAX_OBJECTS]

        objects = np.zeros((MAX_OBJECTS, OBJECT_DIM), dtype=np.float32)
        classes = np.zeros(MAX_OBJECTS, dtype=np.int64)
        mask = np.zeros(MAX_OBJECTS, dtype=np.float32)
        for i, o in enumerate(tracks):
            objects[i] = object_features(o, prof.role_of(o.class_name), px, py)
            classes[i] = class_hash(o.class_name)
            mask[i] = 1.0

        # Player token
        dt = max(1e-3, t - self._prev_t) if self._prev_t is not None else 1.0 / max(1, prof.fps)
        if self._prev_xy is not None:
            vx = (px - self._prev_xy[0]) / dt
            vy = (py - self._prev_xy[1]) / dt
        else:
            vx = vy = 0.0
        hud = record.get('hud') or {}
        hp = float(hud.get('hp', 1.0))
        shield = float(hud.get('shield', 1.0))
        hp_known = 1.0 if float(hud.get('hp_conf', 0.0)) > 0 else 0.0
        hp_delta = hp - self._prev_hp if self._prev_hp is not None else 0.0
        shield_delta = shield - self._prev_shield if self._prev_shield is not None else 0.0
        if hp_delta < -0.002 or shield_delta < -0.002:
            self._last_damage_t = t
        since_damage = (t - self._last_damage_t) if self._last_damage_t is not None else 10.0
        n_enemies = float(objects[:, OBJECT_BASE_DIM + ROLE_INDEX['enemy']].sum())

        player = np.array([
            px, py,
            np.clip(vx, -2, 2), np.clip(vy, -2, 2),
            hp, shield,
            np.clip(hp_delta * 10, -1, 1), np.clip(shield_delta * 10, -1, 1),
            hp_known,
            1.0 if seen else (0.5 if prof.self_fallback == 'center' else 0.0),
            min(since_damage / 10.0, 1.0),
            min(n_enemies / 5.0, 1.0),
        ], dtype=np.float32)
        assert len(player) == PLAYER_DIM

        # Previous action = what actually happened on the previous frame. Read
        # from the previous record NOW (not when it was stepped), because the
        # live agent attaches 'bot_action' to a record after its features are
        # built; a replayed record already has it - so both see the same value.
        prev_action = action_vector(acted_input(self._prev_record), self.keys)

        out = {
            'player': player, 'objects': objects, 'classes': classes, 'mask': mask,
            'prev_action': prev_action, 'tracks': tracks, 'player_xy': (px, py),
        }

        # Update history AFTER building this frame's inputs
        self._prev_t = t
        self._prev_xy = (px, py)
        self._prev_hp = hp
        self._prev_shield = shield
        self._prev_record = record
        return out
