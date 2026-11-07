"""
Training samples from raw V3 sessions.

Each session is replayed through features.FeatureBuilder (same code the live
agent uses), so inputs match runtime exactly. Labels:

  actions  the next CHUNK inputs (mouse xy, left/right, keys) - only from
           frames where a human controlled the game (record / takeover)
  target   the tracked object under the human's next click within 1 s
           (matched by track id), else "none"
  weight   x3 for takeover (DAgger) frames; optional AWR reward weighting

Train/val split is by SESSION (never neighbouring frames on both sides).

Sessions store raw class names and every key pressed, so building them with
the CURRENT game profile applies profile edits (roles, keys) retroactively.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FeatureBuilder, action_vector
from .profile import GameProfile
from .schema import (CHUNK, MAX_OBJECTS, LEARN_FROM, TAKEOVER,
                     TARGET_CLICK_WINDOW_S, TARGET_MAX_DIST, TAKEOVER_WEIGHT)
from .session_io import SessionReader

MAX_FRAME_GAP_S = 0.5   # a bigger gap (pause, lag) breaks an action chunk


@dataclass
class SessionArrays:
    session_id: str
    player: np.ndarray       # [T, P]
    prev_action: np.ndarray  # [T, A]
    objects: np.ndarray      # [T, N, O]
    classes: np.ndarray      # [T, N]      class-name hash
    mask: np.ndarray         # [T, N]
    actions: np.ndarray      # [T, C, A]   future inputs
    action_mask: np.ndarray  # [T, C]      which future steps are valid labels
    target: np.ndarray       # [T]         index into objects, or N = none
    weight: np.ndarray       # [T]
    sample_ok: np.ndarray    # [T] bool    frame usable as a training sample
    t: np.ndarray            # [T]
    reward: np.ndarray       # [T]


def _frame_reward(rec: Dict, prev: Optional[Dict]) -> float:
    """Simple per-frame reward from what the HUD/combat log saw:
    losing HP/shield is bad, kills/loot are good."""
    r = 0.0
    if prev is not None:
        h, ph = rec.get('hud') or {}, prev.get('hud') or {}
        if h.get('hp_conf', 0) > 0 and ph.get('hp_conf', 0) > 0:
            r += 5.0 * (h.get('hp', 1.0) - ph.get('hp', 1.0))
            r += 2.0 * (h.get('shield', 1.0) - ph.get('shield', 1.0))
    ev = rec.get('events') or {}
    r += 1.0 * ev.get('kills', 0) + 0.3 * ev.get('rewards', 0)
    return r


def build_session(session_dir, profile: GameProfile) -> SessionArrays:
    reader = SessionReader(session_dir)
    records = reader.read_all()
    T, C, N = len(records), CHUNK, MAX_OBJECTS
    keys = tuple(profile.keys)
    A = 4 + len(keys)
    fb = FeatureBuilder(profile)

    player, prev_action, objects, classes, mask, track_ids = [], [], [], [], [], []
    for rec in records:
        f = fb.step(rec)
        player.append(f['player']); prev_action.append(f['prev_action'])
        objects.append(f['objects']); classes.append(f['classes']); mask.append(f['mask'])
        track_ids.append([o.track_id for o in f['tracks']])

    t = np.array([float(r.get('t', 0.0)) for r in records], dtype=np.float64)
    ctrl = [r.get('controller') for r in records]
    human = np.array([c in LEARN_FROM for c in ctrl], dtype=bool)
    inputs = [r.get('input') or {} for r in records]
    act = np.stack([action_vector(i, keys) for i in inputs]) if T else np.zeros((0, A), np.float32)

    # Future action chunks (only through contiguous human frames)
    actions = np.zeros((T, C, A), dtype=np.float32)
    action_mask = np.zeros((T, C), dtype=np.float32)
    for i in range(T):
        if not human[i]:
            continue
        for k in range(C):
            j = i + k
            if j >= T or not human[j] or (k > 0 and t[j] - t[j - 1] > MAX_FRAME_GAP_S):
                break
            actions[i, k] = act[j]
            action_mask[i, k] = 1.0

    # Target = object under the next left-click (rising edge) within the window
    lmb = act[:, 2] > 0.5 if T else np.zeros(0, bool)
    rising = np.zeros(T, dtype=bool)
    rising[1:] = lmb[1:] & ~lmb[:-1]
    if T:
        rising[0] = lmb[0]
    clicked_track = np.full(T, -1, dtype=np.int64)   # track id clicked at frame j
    for j in np.nonzero(rising & human)[0]:
        mx, my = act[j, 0], act[j, 1]
        best, best_d = -1, TARGET_MAX_DIST
        for n, tid in enumerate(track_ids[j]):
            d = float(np.hypot(objects[j][n][0] - mx, objects[j][n][1] - my))
            if d < best_d:
                best, best_d = tid, d
        clicked_track[j] = best
    target = np.full(T, N, dtype=np.int64)
    next_click = -1
    for i in range(T - 1, -1, -1):          # scan backwards for "next click"
        if rising[i] and human[i]:
            next_click = i
        if next_click >= 0 and t[next_click] - t[i] <= TARGET_CLICK_WINDOW_S:
            tid = clicked_track[next_click]
            if tid >= 0 and tid in track_ids[i]:
                target[i] = track_ids[i].index(tid)

    weight = np.where(np.array([c == TAKEOVER for c in ctrl]), TAKEOVER_WEIGHT, 1.0).astype(np.float32)
    sample_ok = human & (action_mask[:, 0] > 0)
    reward = np.array([_frame_reward(records[i], records[i - 1] if i else None) for i in range(T)],
                      dtype=np.float32)

    def stack(xs, shape):
        return np.stack(xs).astype(np.float32) if xs else np.zeros((0,) + shape, np.float32)

    from .schema import PLAYER_DIM, OBJECT_DIM
    return SessionArrays(
        session_id=reader.session_id,
        player=stack(player, (PLAYER_DIM,)), prev_action=stack(prev_action, (A,)),
        objects=stack(objects, (N, OBJECT_DIM)),
        classes=np.stack(classes).astype(np.int64) if classes else np.zeros((0, N), np.int64),
        mask=stack(mask, (N,)),
        actions=actions, action_mask=action_mask, target=target, weight=weight,
        sample_ok=sample_ok, t=t, reward=reward)


def apply_awr(s: SessionArrays, gamma: float = 0.98, horizon: int = 100,
              max_weight: float = 20.0) -> None:
    """Advantage-weighted regression: frames followed by better-than-average
    reward get more weight (exp(adv / beta), clipped). In place."""
    T = len(s.reward)
    if T == 0:
        return
    G = np.zeros(T, dtype=np.float64)
    disc = gamma ** np.arange(horizon)
    for i in range(T):
        seg = s.reward[i:i + horizon]
        G[i] = float(np.dot(seg, disc[:len(seg)]))
    ok = s.sample_ok
    if ok.sum() < 2:
        return
    adv = G - G[ok].mean()
    beta = G[ok].std() + 1e-6
    w = np.clip(np.exp(adv / beta), 0.0, max_weight)
    w = w / w[ok].mean()
    s.weight = (s.weight * w).astype(np.float32)


def split_sessions(sessions: Sequence[SessionArrays], val_frac: float = 0.2,
                   gap_s: float = 2.0) -> Tuple[List[Tuple[int, np.ndarray]], List[Tuple[int, np.ndarray]], Dict]:
    """
    Returns (train, val, info) where each list holds (session_index, frame_indices).
    >= 2 sessions: the last ~val_frac of sessions (by start time) are validation.
    1 session: the last val_frac of its frames, with a gap_s gap before it.
    """
    n = len(sessions)
    train, val = [], []
    if n >= 2:
        n_val = max(1, int(round(n * val_frac)))
        for k, s in enumerate(sessions):
            idx = np.nonzero(s.sample_ok)[0]
            (val if k >= n - n_val else train).append((k, idx))
        info = {'mode': 'by_session', 'val_sessions': [s.session_id for s in sessions[n - n_val:]]}
    elif n == 1:
        s = sessions[0]
        idx = np.nonzero(s.sample_ok)[0]
        if len(idx) == 0:
            return [], [], {'mode': 'empty'}
        t_split = s.t[0] + (s.t[-1] - s.t[0]) * (1 - val_frac)
        train.append((0, idx[s.t[idx] < t_split - gap_s]))
        val.append((0, idx[s.t[idx] >= t_split]))
        info = {'mode': 'time_block', 'val_from_t': float(t_split), 'gap_s': gap_s}
    else:
        info = {'mode': 'empty'}
    return train, val, info


class ChunkDataset(Dataset):
    def __init__(self, sessions: Sequence[SessionArrays], parts: Sequence[Tuple[int, np.ndarray]]):
        self.sessions = sessions
        self.index = [(k, int(i)) for k, idx in parts for i in idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        k, i = self.index[n]
        s = self.sessions[k]
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return {
            'player': f(s.player[i]), 'prev_action': f(s.prev_action[i]),
            'objects': f(s.objects[i]), 'classes': f(s.classes[i]), 'mask': f(s.mask[i]),
            'actions': f(s.actions[i]), 'action_mask': f(s.action_mask[i]),
            'target': torch.tensor(s.target[i]), 'weight': torch.tensor(s.weight[i]),
        }
