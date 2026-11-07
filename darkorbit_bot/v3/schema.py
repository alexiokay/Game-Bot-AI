"""
V3 constants. Everything that defines tensor shapes lives here, and the
policy checkpoint stores these values so a model always knows its own shapes.
"""

from pathlib import Path

from ..v2.config import KEYBOARD_KEYS, OBJECT_FEATURE_DIM, DEFAULT_YOLO_MODEL_PATH

# Session format written by session_io.SessionWriter
SESSION_FORMAT_VERSION = 1

# Repo layout (E:/dev/bot/darkorbit_bot/v3/schema.py -> parents[2] = E:/dev/bot)
REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO_ROOT / "darkorbit_bot" / "data" / "v3_sessions"
DEFAULT_POLICY_PATH = REPO_ROOT / "models" / "v3" / "policy.pt"
DEFAULT_YOLO_MODEL = DEFAULT_YOLO_MODEL_PATH

# Recording
RECORD_FPS = 20                 # agent loop / recording rate
VIDEO_SIZE = (960, 540)         # stored video (width, height)

# Inputs
MAX_OBJECTS = 16                # object tokens per frame (nearest first)
OBJECT_DIM = OBJECT_FEATURE_DIM  # 20, from TrackedObject.to_feature_vector
PLAYER_DIM = 12                 # see features.player_features
KEYS = tuple(KEYBOARD_KEYS)     # 32 game keys the policy can press
NUM_KEYS = len(KEYS)
ACTION_DIM = 4 + NUM_KEYS       # mouse x, y, left button, right button, keys
PREV_ACTION_DIM = ACTION_DIM    # previous-action token

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
