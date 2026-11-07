"""
Raw session recording: downscaled video + one JSON line per frame + meta.

    <SESSIONS_DIR>/<session_id>/
        video.mp4      960x540 at ~20 fps (frame_idx in each record points here)
        frames.jsonl   one record per frame (detections, HUD, inputs, controller)
        meta.json      monitor, fps, YOLO model, format version, counts

Files are written as *.tmp and renamed on close(). If the process dies,
frames.jsonl.tmp still holds every complete line and SessionReader reads it.
"""

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .schema import SESSION_FORMAT_VERSION, SESSIONS_DIR, VIDEO_SIZE, RECORD_FPS


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def file_sha1(path: str, limit_bytes: int = 64 * 1024 * 1024) -> Optional[str]:
    """Short hash of a model file (first 64 MB) so sessions name their detector."""
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            h.update(f.read(limit_bytes))
        return h.hexdigest()[:12]
    except OSError:
        return None


class SessionWriter:
    def __init__(self, meta: Dict, root: Path = SESSIONS_DIR, session_id: Optional[str] = None,
                 record_video: bool = True, fps: int = RECORD_FPS):
        self.session_id = session_id or new_session_id()
        self.dir = Path(root) / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.meta.update({
            'format_version': SESSION_FORMAT_VERSION,
            'session_id': self.session_id,
            'fps': fps,
            'video_size': list(VIDEO_SIZE),
            'started': time.time(),
        })
        self._frames_tmp = self.dir / "frames.jsonl.tmp"
        self._video_tmp = self.dir / "video.tmp.mp4"
        self._jsonl = open(self._frames_tmp, 'w', encoding='utf-8')
        self.n_records = 0
        self.n_video_frames = 0
        self.n_dropped = 0
        self._closed = False
        self._lock = threading.Lock()

        # Write meta immediately (as .tmp) so a crashed session is still described
        self._write_json(self.dir / "meta.json.tmp", self.meta)

        self._video_q: Optional[queue.Queue] = None
        self._video_thread = None
        if record_video:
            self._video_q = queue.Queue(maxsize=60)
            self._video_thread = threading.Thread(target=self._video_worker, args=(fps,), daemon=True)
            self._video_thread.start()

    @staticmethod
    def _write_json(path: Path, obj: Dict):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)

    def _video_worker(self, fps: int):
        import cv2
        writer = None
        try:
            while True:
                frame = self._video_q.get()
                if frame is None:
                    break
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(self._video_tmp), fourcc, fps, VIDEO_SIZE)
                if frame.shape[1] != VIDEO_SIZE[0] or frame.shape[0] != VIDEO_SIZE[1]:
                    frame = cv2.resize(frame, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
                writer.write(np.ascontiguousarray(frame[:, :, :3]))
        finally:
            if writer is not None:
                writer.release()

    def add(self, record: Dict, frame: Optional[np.ndarray] = None) -> Dict:
        """Append one frame record (and optionally its screen frame).
        Sets record['frame_idx'] (None if the frame was not stored)."""
        with self._lock:
            if self._closed:
                return record
            record['frame_idx'] = None
            if frame is not None and self._video_q is not None:
                try:
                    self._video_q.put_nowait(frame)
                    record['frame_idx'] = self.n_video_frames
                    self.n_video_frames += 1
                except queue.Full:
                    self.n_dropped += 1  # never block the control loop
            self._jsonl.write(json.dumps(record, separators=(',', ':')) + '\n')
            self._jsonl.flush()
            self.n_records += 1
        return record

    def close(self):
        """Finish video, finalise file names and meta (safe to call twice)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._jsonl.close()
        if self._video_thread is not None:
            self._video_q.put(None)
            self._video_thread.join(timeout=30)
        self.meta.update({
            'ended': time.time(),
            'n_records': self.n_records,
            'n_video_frames': self.n_video_frames,
            'n_dropped_video_frames': self.n_dropped,
            'complete': True,
        })
        os.replace(self._frames_tmp, self.dir / "frames.jsonl")
        if self._video_tmp.exists():
            os.replace(self._video_tmp, self.dir / "video.mp4")
        self._write_json(self.dir / "meta.json.tmp", self.meta)
        os.replace(self.dir / "meta.json.tmp", self.dir / "meta.json")


class SessionReader:
    def __init__(self, session_dir):
        self.dir = Path(session_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            meta_path = self.dir / "meta.json.tmp"
        self.meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        self.session_id = self.meta.get('session_id', self.dir.name)
        frames = self.dir / "frames.jsonl"
        self.frames_path = frames if frames.exists() else self.dir / "frames.jsonl.tmp"
        video = self.dir / "video.mp4"
        self.video_path = video if video.exists() else self.dir / "video.tmp.mp4"
        self.complete = bool(self.meta.get('complete'))

    def records(self) -> Iterator[Dict]:
        """Yield frame records in order; skips a torn last line from a crash."""
        if not self.frames_path.exists():
            return
        with open(self.frames_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def read_all(self) -> List[Dict]:
        return list(self.records())

    def video_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Random-access one stored frame (for re-running a detector later)."""
        import cv2
        if frame_idx is None or not self.video_path.exists():
            return None
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()


def list_sessions(root: Path = SESSIONS_DIR) -> List[Path]:
    """Session folders sorted by id (= start time)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and ((p / "frames.jsonl").exists() or (p / "frames.jsonl.tmp").exists()))
