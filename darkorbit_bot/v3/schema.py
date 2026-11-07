"""
V3 constants. Everything that defines tensor shapes lives here, and the
policy checkpoint stores these values so a model always knows its own shapes.
"""

from pathlib import Path

from ..v2.config import DEFAULT_YOLO_MODEL_PATH, VELOCITY_SCALE

# Session format written by session_io.SessionWriter
SESSION_FORMAT_VERSION = 1

# Repo layout (E:/dev/bot/darkorbit_bot/v3/schema.py -> parents[2] = E:/dev/bot)
REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO_ROOT / "darkorbit_bot" / "data" / "v3_sessions"   # /<game>/<session_id>
POLICY_DIR = REPO_ROOT / "models" / "v3"                     # /<game>/policy.pt
DEFAULT_YOLO_MODEL = DEFAULT_YOLO_MODEL_PATH

# Recording
RECORD_FPS = 20                 # agent loop / recording rate
VIDEO_SIZE = (960, 540)         # stored video (width, height)

# Inputs (game-agnostic; keys and the action size come from the game profile)
MAX_OBJECTS = 16                # object tokens per frame (nearest first)
OBJECT_BASE_DIM = 16            # geometry/motion/tracking part, see features.object_features
OBJECT_DIM = OBJECT_BASE_DIM + 7  # + one-hot over profile.TOKEN_ROLES
PLAYER_DIM = 12                 # see features.FeatureBuilder.step
OBJ_VELOCITY_SCALE = VELOCITY_SCALE

# Outputs
CHUNK = 8                       # future steps predicted per call
REPLAN_EVERY = 4                # run the model every N steps, ensemble chunks

# Labels
TARGET_CLICK_WINDOW_S = 1.0     # target = object under the next click within 1 s
TARGET_MAX_DIST = 0.08          # ... and within this (normalised) distance
TAKEOVER_WEIGHT = 3.0           # DAgger corrections count 3x

# Controllers recorded per frame
HUMAN, BOT, TAKEOVER = "human", "bot", "takeover"
LEARN_FROM = (HUMAN, TAKEOVER)  # frames whose inputs are training labels


def sessions_dir(game: str) -> Path:
    return SESSIONS_DIR / game


def default_policy_path(game: str) -> Path:
    return POLICY_DIR / game / "policy.pt"

# Tracking (V3 settings for v2/perception/tracker.ObjectTracker).
# The tracker's default match_thresh 0.8 needs boxes to overlap >= 80% between
# frames, so any moving object (or the camera scrolling) got a new track every
# frame and left a "lost" ghost for 30 frames. IoU >= 0.1 keeps ids stable at
# fast scroll speeds; only tracks seen in the last 2 frames become tokens.
TRACK_MATCH_IOU = 0.1
TRACK_MAX_MISSED = 2
