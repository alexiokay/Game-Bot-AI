"""
DarkOrbit Bot - Vision Context Analyzer

Uses a local vision-language model (Qwen, LLaVA, etc.) to understand
game context and provide richer training data.

During recording:
- Captures screenshots periodically
- Asks VLM: "What is happening in this game screenshot?"
- Stores the context description alongside movement data

During bot operation:
- Periodically queries VLM for situation assessment
- Uses context to make better decisions

This creates a feedback loop:
1. Record gameplay with VLM context annotations
2. Train policy on (state, action, context) tuples
3. Bot queries VLM and uses context to select actions

Requirements:
- Local VLM server (Ollama with qwen2-vl, llava, etc.)
- OR API access to vision model
"""

import base64
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

try:
    import requests
except ImportError:
    requests = None


@dataclass
class GameContext:
    """Structured game context from VLM analysis"""
    timestamp: float
    raw_description: str

    # Parsed fields
    situation: str  # "combat", "looting", "exploring", "fleeing", "idle"
    threat_level: str  # "none", "low", "medium", "high", "critical"
    nearby_enemies: List[str]
    nearby_items: List[str]
    player_status: str  # "healthy", "damaged", "critical"
    recommended_action: str  # VLM's suggestion

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'raw_description': self.raw_description,
            'situation': self.situation,
            'threat_level': self.threat_level,
            'nearby_enemies': self.nearby_enemies,
            'nearby_items': self.nearby_items,
            'player_status': self.player_status,
            'recommended_action': self.recommended_action
        }


class VisionContextAnalyzer:
    """
    Analyzes game screenshots using a vision-language model.

    Supports multiple backends:
    - Ollama (local, recommended)
    - OpenAI-compatible APIs
    """

    # Prompt template for game analysis
    ANALYSIS_PROMPT = """This is a DarkOrbit space game screenshot. Analyze briefly:

1. situation: combat/looting/exploring/fleeing/idle
2. threat: none/low/medium/high/critical
3. enemies: list visible enemy names
4. items: list visible collectibles
5. health: healthy/damaged/critical
6. action: recommended next action

Reply ONLY with JSON, no other text:
{"situation":"...","threat":"...","enemies":[],"items":[],"health":"...","action":"..."}"""

    def __init__(self,
                 backend: str = "ollama",
                 model: str = "qwen2-vl:7b",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize vision context analyzer.

        Args:
            backend: "ollama" or "openai"
            model: Model name (e.g., "qwen2-vl:7b", "llava:13b")
            base_url: API base URL
        """
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.available = requests is not None

        # Rate limiting - VLM is slow, don't query too often
        self.last_query_time = 0
        self.min_query_interval = 5.0  # Seconds between queries (VLM is slow)

        # Cache recent results
        self.context_cache: List[GameContext] = []
        self.max_cache_size = 100

        if not self.available:
            print("Warning: 'requests' not installed, VLM disabled")

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64"""
        try:
            from PIL import Image
            import io

            # Convert numpy to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # Resize for efficiency (VLMs don't need full res)
            max_size = 800
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size)

            # Encode to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"Image encoding error: {e}")
            return None

    def analyze_screenshot(self, image: np.ndarray) -> Optional[GameContext]:
        """
        Analyze a game screenshot using the VLM.

        Args:
            image: Screenshot as numpy array (RGB)

        Returns:
            GameContext object or None if failed
        """
        if not self.available:
            return None

        # Rate limiting
        now = time.time()
        if now - self.last_query_time < self.min_query_interval:
            return None
        self.last_query_time = now

        # Encode image
        image_b64 = self._encode_image(image)
        if image_b64 is None:
            return None

        try:
            if self.backend == "ollama":
                return self._query_ollama(image_b64)
            else:
                return self._query_openai(image_b64)
        except Exception as e:
            print(f"VLM query error: {e}")
            return None

    def _query_ollama(self, image_b64: str) -> Optional[GameContext]:
        """Query Ollama API"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self.ANALYSIS_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 300
            }
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        raw_text = result.get('response', '')

        return self._parse_response(raw_text)

    def _query_openai(self, image_b64: str) -> Optional[GameContext]:
        """Query OpenAI-compatible API"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            "max_tokens": 300,
            "temperature": 0.3
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        raw_text = result['choices'][0]['message']['content']

        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> Optional[GameContext]:
        """Parse VLM response into GameContext"""
        try:
            # Try to extract JSON from response
            # Handle cases where model wraps JSON in markdown
            text = raw_text.strip()
            if '```' in text:
                # Extract from code block
                start = text.find('{')
                end = text.rfind('}') + 1
                text = text[start:end]

            data = json.loads(text)

            context = GameContext(
                timestamp=time.time(),
                raw_description=raw_text,
                situation=data.get('situation', 'unknown'),
                threat_level=data.get('threat', 'unknown'),
                nearby_enemies=data.get('enemies', []),
                nearby_items=data.get('items', []),
                player_status=data.get('health', 'unknown'),
                recommended_action=data.get('action', '')
            )

            # Cache it
            self.context_cache.append(context)
            if len(self.context_cache) > self.max_cache_size:
                self.context_cache.pop(0)

            return context

        except json.JSONDecodeError:
            # Fallback: create basic context from raw text
            return GameContext(
                timestamp=time.time(),
                raw_description=raw_text,
                situation='unknown',
                threat_level='unknown',
                nearby_enemies=[],
                nearby_items=[],
                player_status='unknown',
                recommended_action=raw_text[:100]
            )

    def get_latest_context(self) -> Optional[GameContext]:
        """Get most recent context analysis"""
        if self.context_cache:
            return self.context_cache[-1]
        return None


class AsyncVisionAnalyzer:
    """
    Async wrapper for VisionContextAnalyzer.
    Runs VLM queries in background thread to avoid blocking game loop.
    """

    def __init__(self, **kwargs):
        self.analyzer = VisionContextAnalyzer(**kwargs)
        self.pending_image = None
        self.latest_context: Optional[GameContext] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start background analysis thread"""
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background analysis"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def submit_image(self, image: np.ndarray):
        """Submit image for async analysis"""
        with self.lock:
            self.pending_image = image.copy()

    def get_context(self) -> Optional[GameContext]:
        """Get latest analysis result"""
        with self.lock:
            return self.latest_context

    def _analysis_loop(self):
        """Background analysis loop"""
        while self.running:
            image = None
            with self.lock:
                if self.pending_image is not None:
                    image = self.pending_image
                    self.pending_image = None

            if image is not None:
                context = self.analyzer.analyze_screenshot(image)
                if context:
                    with self.lock:
                        self.latest_context = context

            time.sleep(0.1)  # Check for new images at 10Hz


def demo():
    """Demo the vision context analyzer"""
    print("\n" + "="*60)
    print("  VISION CONTEXT ANALYZER DEMO")
    print("="*60)

    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        print(f"\n✅ Ollama is running with {len(models)} models")

        # Check for vision models
        vision_models = [m['name'] for m in models if 'vl' in m['name'].lower() or 'llava' in m['name'].lower()]
        if vision_models:
            print(f"   Vision models available: {vision_models}")
        else:
            print("   ⚠️ No vision models found. Install one with:")
            print("      ollama pull qwen2-vl:7b")

    except Exception as e:
        print(f"\n❌ Ollama not available: {e}")
        print("   Install Ollama and run: ollama serve")
        print("   Then: ollama pull qwen2-vl:7b")


if __name__ == "__main__":
    demo()
