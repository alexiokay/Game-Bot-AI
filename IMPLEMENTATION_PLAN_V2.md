# V2 Upgrade Implementation Plan

## Goal
Upgrade the V2 architecture to support complex player tactics and highly efficient data labeling via VLMs.

## User Review Required
> [!IMPORTANT]
> **VLM Cost/Compute**: Running "Sliding Window" analysis on video clips requires significant compute.
> - **Local**: Needs high VRAM (24GB+) for large VLM (e.g., LLaVA/Qwen-VL) to run efficiently.
> - **Cloud**: Gemini 1.5 Flash is recommended for high throughput and low cost.

## Proposed Changes

### 1. VLM Pipeline ("The Tactical Analyst")
Instead of single screenshots, we will send **Frame Sequences** (e.g., 5 frames spread over 2 seconds) to the VLM.

**New File**: `darkorbit_bot/v2/training/vlm_labeler.py`
- **Class**: `BatchSequenceAnalyst`
- **Method**: `analyze_window(frames: List[Image])`
- **Prompt Strategy**:
    - "Watch this 2-second clip."
    - "Extract: Health %, Shield %."
    - "Describe Tactic: 'Kiting', 'Box Collecting', 'Aggressive Push', 'Fleeing'."
    - "Validation: Does the mouse movement match the intent?"

**Parallelism**:
- Use `asyncio` to send 5-10 concurrent requests to the VLM API (if Cloud).
- If Local: Use `batch_size > 1` inference.

### 2. Executor Upgrade (Complex Actions)
The current `[x, y, click, ability]` output is too limited.

**Modify**: [darkorbit_bot/v2/models/executor.py](file:///f:/dev/bot/darkorbit_bot/v2/models/executor.py)
- **New Head Structure**:
    - `mouse_head`: [x, y] (Continuous, 0-1)
    - `click_head`: [no_click, left, right] (Softmax)
    - `hotkey_head`: [none, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, Q, E, R] (Softmax)
- **Config**: Update [ExecutorConfig](file:///f:/dev/bot/darkorbit_bot/v2/config.py#87-106) in [config.py](file:///f:/dev/bot/darkorbit_bot/v2/config.py).

### 3. Data Schema Update
**Modify**: [darkorbit_bot/v2/perception/state_encoder.py](file:///f:/dev/bot/darkorbit_bot/v2/perception/state_encoder.py)
- Add fields for `tactical_label` (string) and `verified_health` (float) to be filled by VLM.

## Verification Plan
### Automated Tests
- Run `test_executor_shapes.py` to ensure new model outputs match expected dimensions.
- Run `test_vlm_mock.py` to verify the JSON parsing of VLM responses.

### Manual Verification
- Run the labeler on a small 30s recording.
- Manually inspect the generated `.json` to ensure "Tactics" descriptions allow for meaningful training.
