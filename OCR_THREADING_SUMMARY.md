# OCR Background Threading Implementation

## Problem Solved
RapidOCR was taking 380-527ms per read on CPU, blocking the main 60fps game loop for ~30 frames and causing mouse movement delays.

## Solution
Moved OCR processing to a background thread so the main loop stays fast (16ms/frame) regardless of OCR speed.

## Changes Made

### File: `darkorbit_bot\v2\perception\gui_detector.py`

#### 1. Thread Infrastructure (lines 297-303)
```python
# Background OCR thread
import threading
import queue
self.ocr_queue = queue.Queue(maxsize=1)  # Only keep latest frame
self.ocr_thread = None
self.ocr_running = False
self.ocr_lock = threading.Lock()  # Protect access to recent_logs
```

#### 2. GPU Detection on Init (lines 309-345)
```python
try:
    self.reader = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
    print("[OCR-INIT] RapidOCR initialized with CUDA GPU acceleration")
except Exception as e:
    self.reader = RapidOCR()
    print(f"[OCR-INIT] RapidOCR running on CPU (GPU error: {e}) - may be slow!")
```

**Key**: Now shows GPU initialization status in console with `print()` statements.

#### 3. Background OCR Worker Thread (lines 355-433)
```python
def _ocr_worker(self):
    """Background OCR worker thread - processes frames from queue."""
    while self.ocr_running:
        # Get frame from queue (non-blocking with timeout)
        log_roi, current_time = self.ocr_queue.get(timeout=0.1)

        # Process OCR (slow part, now in background)
        lines = []
        ocr_start = time.time()

        # ... RapidOCR/EasyOCR/pytesseract processing ...

        # Update results (thread-safe)
        with self.ocr_lock:
            if lines != self.recent_logs:
                self._parse_new_events(lines, current_time)
                # ... update logs with timestamps ...
            self.recent_logs = lines
            self.last_read_time = current_time

        # Log timing every 10 reads
        if self._ocr_timing_counter % 10 == 0:
            print(f"   [OCR-TIMING] {self.ocr_backend} took {ocr_time:.1f}ms | ...")
```

**Key**: Worker runs in background, updates results thread-safely, prints timing info.

#### 4. Non-Blocking `read_logs()` (lines 435-486)
```python
def read_logs(self, frame: np.ndarray, force: bool = False) -> List[str]:
    """
    Non-blocking - submits frames to background thread, returns cached results.
    Main loop stays fast!
    """
    # Submit frame to background OCR if interval elapsed
    if force or time_since_last >= self.read_interval:
        log_roi = frame[y1:y2, x1:x2]

        if log_roi.size > 0:
            log_roi_copy = log_roi.copy()  # Avoid race conditions

            # Drain queue if full, put new frame
            if self.ocr_queue.full():
                self.ocr_queue.get_nowait()
            self.ocr_queue.put_nowait((log_roi_copy, current_time))

    # Return cached results (thread-safe)
    with self.ocr_lock:
        return self.recent_logs.copy()
```

**Key**: Main thread NEVER waits for OCR - just submits frames and reads cached results.

#### 5. Thread Cleanup (lines 543-550)
```python
def stop(self):
    """Stop the background OCR thread."""
    if self.ocr_thread and self.ocr_running:
        print("[OCR-THREAD] Stopping background OCR thread...")
        self.ocr_running = False
        if self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=2.0)
        print("[OCR-THREAD] Background OCR thread stopped")
```

**Key**: Clean shutdown of background thread.

## How It Works

### Before (Blocking):
```
Main Loop (16ms/frame):
  - Capture frame (2ms)
  - YOLO detection (8ms)
  - Policy inference (4ms)
  - OCR read (500ms) ← BLOCKING! Delays 30 frames!
  - Mouse move (2ms)

Total: 516ms per frame = 1.9 fps
```

### After (Non-Blocking):
```
Main Loop (16ms/frame):
  - Capture frame (2ms)
  - YOLO detection (8ms)
  - Policy inference (4ms)
  - OCR submit frame (0.1ms) ← Just copy + queue!
  - Mouse move (2ms)

Background Thread:
  - OCR processing (500ms) ← Runs in parallel!
  - Update cached results

Total: 16ms per frame = 60 fps
```

## Expected Behavior

### On Startup:
```
[OCR-INIT] RapidOCR initialized with CUDA GPU acceleration
[OCR-THREAD] Background OCR thread started
```

OR if GPU fails:
```
[OCR-INIT] RapidOCR running on CPU (GPU error: ...) - may be slow!
[OCR-THREAD] Background OCR thread started
```

### During Running (every ~1 second):
```
   [OCR-TIMING] rapidocr took 379.8ms | Read 0 lines | Lines: []
   [OCR-TIMING] rapidocr took 492.9ms | Read 2 lines | Lines: ['You have received 4 uridium', ...]
```

### On Shutdown:
```
[OCR-THREAD] Stopping background OCR thread...
[OCR-THREAD] Background OCR thread stopped
```

## GPU Acceleration Check

The init message will tell you if GPU is working:
- **GPU working**: `"RapidOCR initialized with CUDA GPU acceleration"`
- **GPU failed**: `"RapidOCR running on CPU (GPU error: ...)"` with error details

If still running on CPU (500ms), the error message will explain why (e.g., CUDA not available, DirectML missing on Windows, etc.).

## Performance Impact

### CPU-bound OCR (current):
- **Before**: 500ms blocking = 1.9 fps
- **After**: 500ms in background = 60 fps (main loop unaffected)

### GPU-accelerated OCR (ideal):
- **Before**: 30-80ms blocking = still noticeable stutter
- **After**: 30-80ms in background = 60 fps (perfect smoothness)

## Thread Safety

- `ocr_queue`: Thread-safe `queue.Queue` for frame submission
- `ocr_lock`: `threading.Lock` protects read/write to `recent_logs`, `recent_logs_with_time`, `event_timestamps`
- Frame copy: `log_roi.copy()` prevents race conditions
- Daemon thread: Auto-terminates when main process exits

## Testing

Run the bot with:
```bash
python test_overfitting.py
```

Watch for:
1. `[OCR-INIT]` message showing GPU status
2. `[OCR-THREAD]` startup message
3. Smooth mouse movement (no more delay!)
4. `[OCR-TIMING]` messages every ~1 second
5. Combat logs being read correctly

## Next Steps

If OCR is still on CPU (slow), check:
1. Is CUDA installed? (`nvidia-smi` to verify)
2. Is rapidocr-onnxruntime built with GPU support?
3. Try installing GPU-enabled ONNX Runtime:
   ```bash
   pip install onnxruntime-gpu
   ```
4. On Windows, may need DirectML support instead of CUDA
