"""
Standalone script to preview VLM corrections without heavy dependencies.
Used by launcher.py to quickly check correction counts.
"""
import json
import sys
from pathlib import Path


def preview_corrections(corrections_dir: str) -> dict:
    """
    Preview what types of corrections exist in a directory.
    Returns dict with counts for each component type.
    """
    corrections_path = Path(corrections_dir)
    if not corrections_path.exists():
        return {'executor': 0, 'strategist': 0, 'tactician': 0, 'total': 0}

    correction_files = list(corrections_path.glob("v2_corrections_*.json"))
    correction_types = {'executor': 0, 'strategist': 0, 'tactician': 0, 'total': 0}

    for file_path in correction_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            corrections = data.get('corrections', [])
            correction_types['total'] += len(corrections)

            for corr in corrections:
                vlm_result = corr.get('vlm_result', {})

                # Detect type of correction
                if 'current_mode_correct' in vlm_result or 'recommended_mode' in vlm_result:
                    correction_types['strategist'] += 1
                if 'quality' in vlm_result or 'correction' in vlm_result:
                    correction_types['executor'] += 1
                if 'target_correct' in vlm_result or 'recommended_target' in vlm_result:
                    correction_types['tactician'] += 1

        except Exception:
            pass

    return correction_types


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preview_vlm_corrections.py <corrections_dir>")
        sys.exit(1)

    corrections_dir = sys.argv[1]
    counts = preview_corrections(corrections_dir)

    # Output as JSON for easy parsing
    print(json.dumps(counts))
