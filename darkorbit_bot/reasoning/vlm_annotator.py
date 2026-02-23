"""
DarkOrbit Bot - VLM Sequence Annotator

Annotates recorded gameplay sequences with vision-language model context.
This runs OFFLINE after recording, not in real-time.

The VLM analyzes screenshots WITH ACTION CONTEXT:
- Screenshot: What's visually happening
- Short-term context: Recent actions (clicks, movements)
- Long-term context: Session patterns

This creates rich training data that understands:
- Not just "what's on screen"
- But "what the player was doing and why"

Usage:
    python vlm_annotator.py                    # Annotate all sequences
    python vlm_annotator.py --session latest   # Annotate latest session
"""

import json
import time
import sys
import base64
from pathlib import Path
from typing import Optional, Dict
from io import BytesIO

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None


class SequenceAnnotator:
    """
    Annotates recorded sequences with VLM context.

    Processes recordings OFFLINE (not real-time) to add rich context
    that helps the policy network learn better.

    Uses actual screenshots + action context for real VLM understanding.
    """

    # Prompt that combines image + context for comprehensive understanding
    VLM_PROMPT_TEMPLATE = """Analyze this DarkOrbit game screenshot with the following ACTION CONTEXT:

RECENT PLAYER ACTIONS:
- Mode: {mode}
- Mouse position: {mouse_pos}
- Enemies visible: {num_enemies}
- Boxes visible: {num_boxes}
- Recent clicks: {recent_clicks}
- Player was: {action_summary}

Based on the screenshot AND the action context above, analyze:

1. situation: What is happening? (combat/looting/exploring/fleeing/idle)
2. threat: Danger level? (none/low/medium/high/critical)
3. quality: Was the player's action appropriate? (good/needs_improvement)
4. reasoning: Brief explanation of why the action was good or bad
5. suggestion: What would be the ideal action here?

Reply ONLY with JSON:
{{"situation":"...","threat":"...","quality":"...","reasoning":"...","suggestion":"..."}}"""

    def __init__(self,
                 model: str = "qwen/qwen3-vl-8b",
                 base_url: str = "http://localhost:1234"):
        """
        Initialize annotator.

        Args:
            model: VLM model name (for LM Studio)
            base_url: LM Studio API URL
        """
        self.model = model
        self.base_url = base_url
        self.available = requests is not None and Image is not None

        if not self.available:
            print("Warning: 'requests' or 'PIL' not installed, VLM disabled")

        self.annotated_count = 0
        self.error_count = 0
        self.skipped_count = 0

    def _encode_image(self, image_path: Path) -> Optional[str]:
        """Load and encode image to base64"""
        try:
            img = Image.open(image_path)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"   Error encoding image: {e}")
            return None

    def _query_vlm(self, image_b64: str, context: Dict) -> Optional[Dict]:
        """
        Query LM Studio VLM with image + context.

        This is the actual VLM call that understands both the visual
        and the action context together.
        """
        # Build action summary from context
        action_summary = "unknown"
        if context.get('recent_clicks', 0) > 3:
            action_summary = "actively clicking targets"
        elif context.get('num_enemies', 0) > 0:
            action_summary = "observing enemies"
        elif context.get('num_boxes', 0) > 0:
            action_summary = "near collectibles"
        else:
            action_summary = "exploring/moving"

        prompt = self.VLM_PROMPT_TEMPLATE.format(
            mode=context.get('mode', 'PASSIVE'),
            mouse_pos=context.get('mouse_pos', (0, 0)),
            num_enemies=context.get('num_enemies', 0),
            num_boxes=context.get('num_boxes', 0),
            recent_clicks=context.get('recent_clicks', 0),
            action_summary=action_summary
        )

        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }],
                "max_tokens": 400,
                "temperature": 0.3
            }

            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            raw_text = result['choices'][0]['message']['content']

            # Parse JSON response
            text = raw_text.strip()
            if '```' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                text = text[start:end]

            return json.loads(text)

        except requests.exceptions.ConnectionError:
            print("   LM Studio not running? Start it and load a vision model.")
            return None
        except json.JSONDecodeError:
            # Return raw text if JSON parsing fails
            return {'raw_response': raw_text, 'quality': 'unknown'}
        except Exception as e:
            print(f"   VLM query error: {e}")
            return None

    def annotate_sequence(self, sequence_path: Path) -> bool:
        """
        Annotate a sequence using its saved screenshots + context.

        This actually loads screenshots and queries LM Studio!
        """
        try:
            with open(sequence_path, 'r') as f:
                data = json.load(f)

            # Skip if already annotated with VLM (not just data inference)
            existing = data.get('vlm_context', {})
            if existing.get('source') == 'vlm_image':
                print(f"   Skipping {sequence_path.name} (already VLM annotated)")
                self.skipped_count += 1
                return False

            # Check for screenshots - REQUIRED for VLM annotation
            screenshots = data.get('screenshots', [])
            if not screenshots:
                print(f"   SKIP: No screenshots for {sequence_path.name}")
                print(f"         Re-record with latest filtered_recorder.py to capture screenshots")
                self.skipped_count += 1
                return False

            # Find screenshots directory
            screenshots_dir = sequence_path.parent / "screenshots"
            if not screenshots_dir.exists():
                print(f"   Screenshots dir not found for {sequence_path.name}")
                return False

            # Analyze middle screenshot (most representative)
            mid_idx = len(screenshots) // 2
            ss_info = screenshots[mid_idx]

            image_path = screenshots_dir / ss_info['image']
            context_path = screenshots_dir / ss_info['context']

            if not image_path.exists():
                print(f"   Screenshot not found: {image_path}")
                return False

            # Load context
            context = {}
            if context_path.exists():
                with open(context_path, 'r') as f:
                    context = json.load(f)

            # Encode image
            image_b64 = self._encode_image(image_path)
            if not image_b64:
                return False

            # Query VLM
            print(f"   Querying VLM for {sequence_path.name}...")
            vlm_result = self._query_vlm(image_b64, context)

            if vlm_result:
                # Use VLM result directly (no fake inference)
                vlm_context = {
                    **vlm_result,
                    'source': 'vlm_image',
                    'screenshot_used': ss_info['image'],
                    'action_context': context,
                    'mode': data.get('mode', 'PASSIVE'),
                    'label': data.get('label', 'UNKNOWN')
                }

                data['vlm_context'] = vlm_context
                data['vlm_annotated'] = True
                data['vlm_timestamp'] = time.time()

                with open(sequence_path, 'w') as f:
                    json.dump(data, f, indent=2)

                self.annotated_count += 1
                quality = vlm_result.get('quality', 'unknown')
                print(f"   Annotated: {sequence_path.name} (quality: {quality})")
                return True

        except Exception as e:
            print(f"   Error annotating {sequence_path.name}: {e}")
            self.error_count += 1

        return False

    def annotate_session(self, session_dir: Path) -> Dict:
        """
        Annotate all sequences in a session.

        Returns:
            Stats dict with annotated/skipped/error counts
        """
        sequences = list(session_dir.glob("sequence_*.json"))
        print(f"\n Processing {len(sequences)} sequences in {session_dir.name}")

        # Check for screenshots - REQUIRED for real VLM annotation
        screenshots_dir = session_dir / "screenshots"
        has_screenshots = screenshots_dir.exists() and any(screenshots_dir.glob("*.jpg"))
        if has_screenshots:
            print(f"   Found screenshots directory - will query LM Studio VLM")
        else:
            print(f"   WARNING: No screenshots found!")
            print(f"   Re-record with latest filtered_recorder.py to capture screenshots")

        for seq_path in sequences:
            self.annotate_sequence(seq_path)
            time.sleep(1.0)  # Delay between VLM queries to avoid overload

        return {
            'total': len(sequences),
            'annotated': self.annotated_count,
            'skipped': self.skipped_count,
            'errors': self.error_count
        }

    def annotate_all(self, recordings_dir: str = "data/recordings") -> Dict:
        """
        Annotate all sessions in the recordings directory.
        """
        recordings_path = Path(recordings_dir)
        if not recordings_path.exists():
            print(f"Recordings directory not found: {recordings_dir}")
            return {'total': 0, 'annotated': 0, 'skipped': 0, 'errors': 0}

        sessions = list(recordings_path.glob("session_*"))
        print(f"\nFound {len(sessions)} sessions to process")

        total_stats = {'total': 0, 'annotated': 0, 'skipped': 0, 'errors': 0}

        for session_dir in sessions:
            stats = self.annotate_session(session_dir)
            total_stats['total'] += stats['total']
            total_stats['annotated'] += stats['annotated']
            total_stats['skipped'] += stats['skipped']
            total_stats['errors'] += stats['errors']

        return total_stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Annotate recorded sequences with VLM context')
    parser.add_argument('--session', type=str, default=None,
                        help='Session to annotate (or "latest")')
    parser.add_argument('--model', type=str, default='qwen/qwen3-vl-8b',
                        help='VLM model name in LM Studio')
    parser.add_argument('--url', type=str, default='http://localhost:1234',
                        help='LM Studio API URL')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  VLM SEQUENCE ANNOTATOR")
    print("="*60)
    print(f"\n  Model: {args.model}")
    print(f"  URL: {args.url}")
    print("\n  This annotator uses REAL screenshots + action context")
    print("  to understand what the player was doing and why.")
    print("-"*60)

    # Check LM Studio connection
    if requests:
        try:
            test_url = f"{args.url}/v1/models"
            resp = requests.get(test_url, timeout=5)
            if resp.status_code == 200:
                models = resp.json().get('data', [])
                print(f"\n  LM Studio connected! {len(models)} model(s) available")
                for m in models[:3]:
                    print(f"    - {m.get('id', 'unknown')}")
            else:
                print(f"\n  Warning: LM Studio returned status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print("\n  Warning: Cannot connect to LM Studio!")
            print(f"  Make sure LM Studio is running at {args.url}")
            print("  And load a vision model (e.g., qwen/qwen3-vl-8b)")
    print("-"*60)

    annotator = SequenceAnnotator(
        model=args.model,
        base_url=args.url
    )

    recordings_dir = Path("data/recordings")

    if args.session:
        if args.session == 'latest':
            sessions = sorted(recordings_dir.glob("session_*"))
            if sessions:
                session_dir = sessions[-1]
            else:
                print("No sessions found!")
                return
        else:
            session_dir = recordings_dir / args.session

        stats = annotator.annotate_session(session_dir)
    else:
        stats = annotator.annotate_all()

    print("\n" + "="*60)
    print("  ANNOTATION COMPLETE")
    print("="*60)
    print(f"   Total sequences: {stats['total']}")
    print(f"   VLM annotated: {stats['annotated']}")
    print(f"   Skipped (already done): {stats['skipped']}")
    print(f"   Errors: {stats['errors']}")
    print("\n  The VLM annotations include:")
    print("    - situation (combat/looting/exploring/etc)")
    print("    - threat level")
    print("    - quality assessment (was player action good?)")
    print("    - reasoning (why good/bad)")
    print("    - suggestion (ideal action)")
    print("="*60)


if __name__ == "__main__":
    main()
