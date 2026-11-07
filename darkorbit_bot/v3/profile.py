"""
Game profiles: everything game-specific lives in games/<name>/profile.yaml.

The V3 learning core (object tokens -> one policy, raw sessions, takeover
corrections) is game-agnostic. A profile tells it:

  detector   which YOLO weights to run, and where "our" character is if not seen
  classes    detector class name -> ROLE (first matching rule wins, fnmatch)
  hud        optional hp/shield bars (screen region + HSV colour range)
  controls   mouse mode, which keys the policy may press, button semantics
  recording  fps and stored video size

Roles are fixed across games (TOKEN_ROLES below), so object tokens mean the
same thing in every game; a learned per-class embedding adds the detail.
"""

import fnmatch
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .schema import REPO_ROOT

GAMES_DIR = REPO_ROOT / "games"
DEFAULT_GAME = "darkorbit"

# Object roles shared by every game ('self' = our character, not a token;
# 'ignore' = dropped). Order defines the one-hot in object tokens.
TOKEN_ROLES: Tuple[str, ...] = ('enemy', 'loot', 'player', 'ally', 'hazard', 'interactive', 'other')
ALL_ROLES = TOKEN_ROLES + ('self', 'ignore')

CLASS_HASH_BUCKETS = 64     # class names -> learned embedding via a stable hash

MOUSE_MODES = ('absolute',)  # 'relative' (FPS mouse-look) is not supported yet
BUTTON_MODES = ('hold', 'click')

# Keys that both input capture (VK codes) and the bot's keyboard (scan codes)
# support today; a profile may use any subset.
SUPPORTED_KEYS: Tuple[str, ...] = (
    'ctrl', 'shift', 'alt', 'space', 'tab', 'esc',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    'q', 'w', 'e', 'r', 't', 'a', 's', 'd', 'f', 'g',
    'z', 'x', 'c', 'v', 'b', 'j',
)


def class_hash(name: str) -> int:
    """Stable bucket for a class name (same in every run and on every machine)."""
    return zlib.crc32(name.encode('utf-8')) % CLASS_HASH_BUCKETS


@dataclass
class ClassRule:
    role: str
    patterns: List[str]

    def matches(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, p) for p in self.patterns)


@dataclass
class HudBar:
    region: Tuple[float, float, float, float]
    hsv_low: Tuple[int, int, int]
    hsv_high: Tuple[int, int, int]
    alt_hsv_low: Optional[Tuple[int, int, int]] = None
    alt_hsv_high: Optional[Tuple[int, int, int]] = None


@dataclass
class GameProfile:
    name: str
    description: str = ""
    detector_weights: str = "best.pt"
    detector_confidence: float = 0.3
    self_fallback: str = "center"            # center | none
    class_rules: List[ClassRule] = field(default_factory=list)
    default_role: str = "other"
    hp_bar: Optional[HudBar] = None
    shield_bar: Optional[HudBar] = None
    mouse_mode: str = "absolute"
    left_button: str = "hold"
    right_button: str = "hold"
    keys: Tuple[str, ...] = ()
    fps: int = 20
    video_size: Tuple[int, int] = (960, 540)
    source: Optional[str] = None             # file it was loaded from
    raw: Dict = field(default_factory=dict)  # the YAML as loaded (stored in sessions)

    # ------------------------------------------------------------------ roles
    def role_of(self, class_name: str) -> str:
        cache = self.__dict__.setdefault('_role_cache', {})
        r = cache.get(class_name)
        if r is None:
            r = next((rule.role for rule in self.class_rules if rule.matches(class_name)), self.default_role)
            cache[class_name] = r
        return r

    @property
    def num_keys(self) -> int:
        return len(self.keys)

    @property
    def action_dim(self) -> int:  # mouse x, y, left, right, keys
        return 4 + len(self.keys)

    def detector_path(self) -> str:
        p = Path(self.detector_weights)
        return str(p if p.is_absolute() else REPO_ROOT / p)

    def compatible_with(self, other: 'GameProfile') -> Optional[str]:
        """None if sessions/policies of both profiles can be mixed, else why not."""
        if tuple(self.keys) != tuple(other.keys):
            return f"different key lists ({len(self.keys)} vs {len(other.keys)})"
        if self.mouse_mode != other.mouse_mode:
            return f"different mouse mode ({self.mouse_mode} vs {other.mouse_mode})"
        return None


# ---------------------------------------------------------------------- load

def _bar(d: Optional[Dict], where: str) -> Optional[HudBar]:
    if not d:
        return None
    try:
        lo, hi = d['hsv']
        alt = d.get('alt_hsv')
        return HudBar(region=tuple(float(v) for v in d['region']),
                      hsv_low=tuple(int(v) for v in lo), hsv_high=tuple(int(v) for v in hi),
                      alt_hsv_low=tuple(int(v) for v in alt[0]) if alt else None,
                      alt_hsv_high=tuple(int(v) for v in alt[1]) if alt else None)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{where}: needs region: [x1, y1, x2, y2] and hsv: [[h,s,v], [h,s,v]] ({e})")


def profile_from_dict(d: Dict, source: Optional[str] = None) -> GameProfile:
    where = source or d.get('name', 'profile')
    det = d.get('detector') or {}
    ctl = d.get('controls') or {}
    rec = d.get('recording') or {}
    hud = (d.get('hud') or {}).get('bars') or {}

    rules = []
    for i, r in enumerate(d.get('classes') or []):
        role = r.get('role')
        if role not in ALL_ROLES:
            raise ValueError(f"{where}: classes[{i}] role '{role}' - use one of {', '.join(ALL_ROLES)}")
        pats = r.get('match') or []
        pats = [pats] if isinstance(pats, str) else [str(p) for p in pats]
        rules.append(ClassRule(role, pats))

    default_role = d.get('default_role', 'other')
    if default_role not in ALL_ROLES:
        raise ValueError(f"{where}: default_role '{default_role}' - use one of {', '.join(ALL_ROLES)}")

    keys = tuple(str(k).lower() for k in (ctl.get('keys') or ()))
    bad = [k for k in keys if k not in SUPPORTED_KEYS]
    if bad:
        raise ValueError(f"{where}: unsupported keys {bad}. Supported: {', '.join(SUPPORTED_KEYS)}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{where}: duplicate keys in controls.keys")

    mouse = ctl.get('mouse', 'absolute')
    if mouse not in MOUSE_MODES:
        raise ValueError(f"{where}: controls.mouse '{mouse}' is not supported yet "
                         f"(supported: {', '.join(MOUSE_MODES)})")
    lb, rb = ctl.get('left_button', 'hold'), ctl.get('right_button', 'hold')
    for name, v in (('left_button', lb), ('right_button', rb)):
        if v not in BUTTON_MODES:
            raise ValueError(f"{where}: controls.{name} '{v}' - use one of {', '.join(BUTTON_MODES)}")

    fallback = det.get('self_fallback', 'center')
    if fallback not in ('center', 'none'):
        raise ValueError(f"{where}: detector.self_fallback must be 'center' or 'none'")

    return GameProfile(
        name=str(d.get('name') or Path(source or 'game').parent.name),
        description=str(d.get('description', '')),
        detector_weights=str(det.get('weights', 'best.pt')),
        detector_confidence=float(det.get('confidence', 0.3)),
        self_fallback=fallback,
        class_rules=rules, default_role=default_role,
        hp_bar=_bar(hud.get('hp'), f"{where}: hud.bars.hp"),
        shield_bar=_bar(hud.get('shield'), f"{where}: hud.bars.shield"),
        mouse_mode=mouse, left_button=lb, right_button=rb, keys=keys,
        fps=int(rec.get('fps', 20)),
        video_size=tuple(int(v) for v in rec.get('video_size', (960, 540))),
        source=source, raw=d)


def load_profile(name_or_path: str = DEFAULT_GAME) -> GameProfile:
    """'darkorbit' -> games/darkorbit/profile.yaml, or a path to a .yaml file."""
    p = Path(name_or_path)
    if p.suffix.lower() not in ('.yaml', '.yml'):
        p = GAMES_DIR / name_or_path / "profile.yaml"
    if not p.exists():
        avail = sorted(d.name for d in GAMES_DIR.iterdir() if (d / 'profile.yaml').exists()) if GAMES_DIR.exists() else []
        raise FileNotFoundError(f"No game profile at {p}. Available: {', '.join(avail) or 'none'}")
    with open(p, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f) or {}
    return profile_from_dict(d, source=str(p))


def list_games() -> List[str]:
    if not GAMES_DIR.exists():
        return []
    return sorted(d.name for d in GAMES_DIR.iterdir()
                  if (d / 'profile.yaml').exists() and not d.name.startswith('_'))
