"""
V3 decision overlay: what the policy saw and decided, drawn over the screen.

    build_state()      record + features + PolicyRunner.last_debug -> JSON-able dict
    draw_overlay()     state + frame -> annotated image (same for live and replay)
    StateBroadcaster   agent side: serves the latest state + JPEG frame on 127.0.0.1
    StateClient        viewer side: connects and yields (state, frame)

Drawn: every detection (tokens coloured by role, ignored/self thin), the
policy's target probability per object, the chosen target, the predicted
mouse path, your cursor, buttons/keys, HUD and who is in control.
"""

import json
import queue
import socket
import struct
import threading
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .profile import GameProfile, TOKEN_ROLES

# BGR colours per role
ROLE_COLORS = {
    'enemy': (60, 60, 235), 'loot': (60, 210, 255), 'player': (230, 160, 60),
    'ally': (90, 220, 90), 'hazard': (200, 80, 220), 'interactive': (240, 220, 120),
    'other': (180, 180, 180), 'self': (255, 255, 0), 'ignore': (110, 110, 110),
}
CONTROLLER_COLORS = {'bot': (60, 60, 235), 'takeover': (0, 200, 255), 'human': (90, 220, 90),
                     'paused': (150, 150, 150)}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def build_state(record: Dict, feats: Dict, profile: GameProfile, debug: Optional[Dict]) -> Dict:
    """Everything the overlay draws, as plain JSON types."""
    probs = (debug or {}).get('target_probs') or {}
    objects = []
    for o in feats.get('tracks') or []:
        tid = int(o.track_id)
        objects.append({'tid': tid, 'cls': o.class_name, 'role': profile.role_of(o.class_name),
                        'x': round(float(o.x), 4), 'y': round(float(o.y), 4),
                        'w': round(float(o.width), 4), 'h': round(float(o.height), 4),
                        'p': round(float(probs.get(tid, 0.0)), 3) if probs else None,
                        'lost': o.state == 'lost'})
    passive = []   # self / ignored detections, drawn thin
    for d in record.get('dets') or []:
        role = profile.role_of(d['c'])
        if role in ('self', 'ignore'):
            passive.append({'cls': d['c'], 'role': role, 'b': d['b']})
    px, py = feats.get('player_xy', (0.5, 0.5))
    return {
        't': record.get('t'), 'frame_idx': record.get('frame_idx'),
        'controller': record.get('controller'), 'hud': record.get('hud'),
        'input': record.get('input'),
        'bot_action': record.get('bot_action') or record.get('bot_suggestion'),
        'bot_is_suggestion': 'bot_suggestion' in record and 'bot_action' not in record,
        'player_xy': [round(float(px), 4), round(float(py), 4)],
        'objects': objects, 'passive': passive,
        'target_tid': (debug or {}).get('target_track_id'),
        'none_prob': (debug or {}).get('none_prob'),
        'mouse_path': (debug or {}).get('mouse_path') or [],
        'button_probs': (debug or {}).get('button_probs'),
        'keys': list(profile.keys), 'game': profile.name,
    }


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _pt(x, y, W, H) -> Tuple[int, int]:
    return int(round(x * W)), int(round(y * H))


def draw_overlay(frame: Optional[np.ndarray], state: Dict, size: Tuple[int, int] = (960, 540),
                 show_passive: bool = True) -> np.ndarray:
    import cv2
    W, H = size
    if frame is None:
        img = np.zeros((H, W, 3), np.uint8)
    else:
        img = frame if (frame.shape[1], frame.shape[0]) == (W, H) else cv2.resize(frame, (W, H))
        img = np.ascontiguousarray(img[:, :, :3]).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Self / ignored detections (thin)
    if show_passive:
        for d in state.get('passive') or []:
            x1, y1, x2, y2 = d['b']
            c = ROLE_COLORS.get(d['role'], (110, 110, 110))
            cv2.rectangle(img, _pt(x1, y1, W, H), _pt(x2, y2, W, H), c, 1)

    # Object tokens with target probability
    tgt = state.get('target_tid')
    for o in state.get('objects') or []:
        c = ROLE_COLORS.get(o['role'], (180, 180, 180))
        p1 = _pt(o['x'] - o['w'] / 2, o['y'] - o['h'] / 2, W, H)
        p2 = _pt(o['x'] + o['w'] / 2, o['y'] + o['h'] / 2, W, H)
        is_t = tgt is not None and o['tid'] == tgt
        cv2.rectangle(img, p1, p2, c, 3 if is_t else 1, lineType=cv2.LINE_AA)
        if o.get('p') is not None:
            # probability bar above the box
            bw = max(1, int((p2[0] - p1[0]) * min(1.0, o['p'])))
            cv2.rectangle(img, (p1[0], p1[1] - 6), (p1[0] + bw, p1[1] - 2), c, -1)
            label = f"{o['role']} {o['p'] * 100:.0f}%"
        else:
            label = o['role']
        cv2.putText(img, label, (p1[0], p1[1] - 9), font, 0.4, c, 1, cv2.LINE_AA)
        if is_t:
            cv2.putText(img, "TARGET", (p1[0], p2[1] + 14), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Our position
    px, py = state.get('player_xy') or (0.5, 0.5)
    cv2.circle(img, _pt(px, py, W, H), 6, ROLE_COLORS['self'], 1, cv2.LINE_AA)

    # Predicted mouse path (bot) and the actual / human cursor
    path = state.get('mouse_path') or []
    if path:
        pts = [_pt(x, y, W, H) for x, y in path]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (60, 60, 235), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 3 if i else 5, (60, 60, 235), -1, cv2.LINE_AA)
    ba = state.get('bot_action')
    if ba:
        bp = _pt(ba['mx'], ba['my'], W, H)
        cv2.drawMarker(img, bp, (60, 60, 235), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    inp = state.get('input')
    if inp and state.get('controller') in ('human', 'takeover', 'paused'):
        hp_ = _pt(inp['mx'], inp['my'], W, H)
        cv2.drawMarker(img, hp_, (90, 220, 90), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    # Status bar
    bar_h = 44
    cv2.rectangle(img, (0, H - bar_h), (W, H), (25, 25, 25), -1)
    ctrl = state.get('controller') or '?'
    cv2.putText(img, ctrl.upper(), (8, H - 26), font, 0.55, CONTROLLER_COLORS.get(ctrl, (200, 200, 200)), 2, cv2.LINE_AA)
    hud = state.get('hud') or {}
    hud_txt = (f"HP {hud.get('hp', 0) * 100:.0f}%" + ("" if hud.get('hp_conf', 0) else "?") +
               f"  SH {hud.get('shield', 0) * 100:.0f}%" if hud else "HUD off")
    t = state.get('t')
    cv2.putText(img, f"{hud_txt}   objects {len(state.get('objects') or [])}   "
                     f"t {t:.1f}s" if t is not None else hud_txt, (130, H - 26), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    # Buttons / keys the bot wants (prob > 0.5) and what you pressed
    keys = state.get('keys') or []
    bp = state.get('button_probs')
    who = "BOT (suggestion)" if state.get('bot_is_suggestion') else "BOT"
    if bp:
        names = ['LMB', 'RMB'] + list(keys)
        want = [f"{n}" for n, p in zip(names, bp) if p > 0.5]
        cv2.putText(img, f"{who}: {' '.join(want) or '-'}" +
                    (f"   none {state['none_prob'] * 100:.0f}%" if state.get('none_prob') is not None else ""),
                    (8, H - 8), font, 0.45, (60, 60, 235), 1, cv2.LINE_AA)
    if inp:
        you = (['LMB'] if inp.get('lmb') else []) + (['RMB'] if inp.get('rmb') else []) + list(inp.get('keys') or [])
        cv2.putText(img, f"YOU: {' '.join(you) or '-'}", (W // 2, H - 8), font, 0.45, (90, 220, 90), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# Live transport: length-prefixed [json][jpeg] messages over localhost TCP
# --------------------------------------------------------------------------

def _encode(state: Dict, frame: Optional[np.ndarray], width: int = 960, quality: int = 70) -> bytes:
    import cv2
    jpg = b''
    if frame is not None:
        h, w = frame.shape[:2]
        if w != width:
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(frame[:, :, :3]), [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg = buf.tobytes() if ok else b''
    js = json.dumps(state, separators=(',', ':')).encode('utf-8')
    return struct.pack('!II', len(js), len(jpg)) + js + jpg


class StateBroadcaster:
    """Serves the newest overlay state to any number of local viewers. Each
    viewer has its own sender thread and keeps only the latest message, so a
    slow viewer never blocks the agent and never receives a torn message."""

    def __init__(self, host: str = '127.0.0.1', port: int = 9998):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(4)
        self._srv.settimeout(0.5)
        self.port = self._srv.getsockname()[1]
        self._clients: List[Tuple[socket.socket, queue.Queue]] = []
        self._lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while not self._closed:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            q: queue.Queue = queue.Queue(maxsize=1)
            with self._lock:
                self._clients.append((conn, q))
            threading.Thread(target=self._send_loop, args=(conn, q), daemon=True).start()

    def _send_loop(self, conn: socket.socket, q: queue.Queue):
        try:
            while not self._closed:
                msg = q.get()
                if msg is None:
                    break
                conn.sendall(msg)            # blocking, whole message or error
        except OSError:
            pass
        finally:
            with self._lock:
                self._clients = [(c, qq) for c, qq in self._clients if c is not conn]
            try:
                conn.close()
            except OSError:
                pass

    def has_clients(self) -> bool:
        return bool(self._clients)

    def publish(self, state: Dict, frame: Optional[np.ndarray] = None):
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        msg = _encode(state, frame)
        for _, q in clients:
            try:                              # keep only the newest message
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def close(self):
        self._closed = True
        with self._lock:
            clients = list(self._clients)
        for conn, q in clients:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            self._srv.close()
        except OSError:
            pass


class StateClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 9998, timeout: float = 5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self):
        self.sock = socket.create_connection(self.addr, timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("agent closed the connection")
            buf += chunk
        return bytes(buf)

    def receive(self) -> Tuple[Dict, Optional[np.ndarray]]:
        import cv2
        js_len, jpg_len = struct.unpack('!II', self._recv_exact(8))
        state = json.loads(self._recv_exact(js_len).decode('utf-8'))
        frame = None
        if jpg_len:
            frame = cv2.imdecode(np.frombuffer(self._recv_exact(jpg_len), np.uint8), cv2.IMREAD_COLOR)
        return state, frame

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
