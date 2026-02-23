# V2 Hierarchical Training Workflow

This document outlines the optimal training process for the V2 Architecture.

## 1. The Core Philosophy: "Behavioral Cloning via VLM Insight"

Game bots fail when they just copy clicks. They need to copy **behavior**.
We use a **VLM (Vision-Language Model)** not just as a labeler, but as a "Coach" that watches your gameplay and explains *why* you did what you did.

## 2. Advanced VLM Pipeline: "The Sliding Window"

To capture tactics (motion, aggression, retreat), a single screenshot is not enough.

### The Technique
*   **Input**: A sequence of **5 frames** uniformly sampled from a 2-second window.
*   **Prompt**: "Watch this 2-second clip. The player is moving the mouse to [X,Y]. Describe the tactic."
*   **Output (JSON)**:
    ```json
    {
      "health": 85,
      "tactic": "Kiting",
      "description": "Player is maintaining distance from Devo while firing.",
      "intent": "FIGHT"
    }
    ```

### Throughput Strategy (Parallelism)
We don't wait for one request to finish.
1.  **Slice** the recording into 2-second windows (overlapping).
2.  **Batch** send 10-20 requests to Gemini Flash (Cloud) or parallel local batches.
3.  **Merge** the results back into the `.json` recording file.

---

## 3. The Upgrade Plan

### Step 1: Executor Upgrade (Hardware)
The current Executor is too simple. We are upgrading it to support a **Full MMO Hotbar**.
*   **Old**: `[x, y, click, ability_val]`
*   **New**: `[x, y, click_type, hotkey_id]`
    *   `hotkey_id`: Softmax over [1-9, Q, E, R, Space, Ctrl...]

### Step 2: VLM "Coach" Implementation
We will build `v2/training/vlm_labeler.py`.
*   **Input**: Directory of `filtered_recorder` sessions.
*   **Process**:
    *   Reads `metadata.json` to find interesting sequences (combat).
    *   Extracts frame windows.
    *   Queries VLM.
    *   Saves `ground_truth.json`.

### Step 3: Layered Training
1.  **Strategist**: Trains on VLM's "Intent" labels (FIGHT vs FLEE).
2.  **Tactician**: Trains on VLM's "Description" (Target selection in crowds).
3.  **Executor**: Trains on raw mouse data, conditioned on the Strategist's goal.

---

### Step 4: The "Causality Bridge" (Addressing the 2-second limit)
You might ask: *"How does a 2-second VLM window understand 10-second consequences?"*
**Answer**: It doesn't need to. That's the **Strategist's** job.

*   **VLM (The Labeler)**: Looks at T=60s. Sees you running away. Labels it "FLEE". It describes the *result*.
*   **Strategist (The Student)**: Looks at T=0s to T=60s (History). Sees you took heavy damage at T=50s.
*   **The Training**: The Strategist learns *"When I take damage at T=50, the correct label at T=60 is 'FLEE'."*
*   **Conclusion**: We distill the VLM's *short-term recognition* into the Strategist's *long-term memory*.

---

## 5. Is this "Hallucinating"?
**No.** This is State-of-the-Art (SOTA) for agentic systems.
*   **DeepMind/OpenAI** use "Language-Goal-Conditioned Policies" exactly like this.
*   **VLA (Vision-Language-Action)** models are the current frontier.
*   By treating the VLM as a teacher, you bypass the need for complex, brittle heuristic code ("if health < 10%..."). The VLM *sees* you running away and labels it "FLEE", and the Strategist learns that pattern naturally.

This path is ambitious but **realistic** and provides the highest quality ceiling for your bot.
