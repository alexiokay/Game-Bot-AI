"""
V2 Meta-Learner - Hierarchical Policy Self-Improvement

Analyzes VLM corrections specific to V2's hierarchical architecture:

STRATEGIST LEVEL (1Hz):
- Mode selection errors (FIGHT vs FLEE vs LOOT patterns)
- Strategic decision patterns over time
- Health-based mode switching accuracy

TACTICIAN LEVEL (10Hz):
- Target selection errors (which targets are wrong, why)
- Target position patterns (clustered, systematic bias)
- Distance/approach optimization issues

EXECUTOR LEVEL (60Hz):
- Mouse positioning errors (systematic bias, clicking out of screen)
- Spatial clustering detection (overfitting to one region)
- Click timing and positioning accuracy

Key V2-specific features:
1. Hierarchical correction analysis (separate for each policy level)
2. Overfitting detection (executor always clicking same region)
3. Positional bias diagnosis (training data from one map area)
4. Actionable fixes for systematic errors

Usage:
    # After bot session ends:
    from darkorbit_bot.v2.vlm.vlm_meta_learner_v2 import MetaLearnerV2
    learner = MetaLearnerV2()
    learner.analyze_session()
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict


class MetaLearnerV2:
    """
    V2-specific meta-learner for hierarchical policy improvements.

    Analyzes corrections at three levels:
    - Strategist: Mode selection (FIGHT/FLEE/LOOT/EXPLORE/CAUTIOUS)
    - Tactician: Target selection from tracked objects
    - Executor: Mouse positioning and clicking
    """

    # Where V2 VLM corrections are saved
    CORRECTIONS_DIR = Path(__file__).parent.parent.parent / "data" / "vlm_corrections_v2"

    # Where suggestions are saved
    SUGGESTIONS_FILE = Path(__file__).parent.parent.parent / "data" / "vlm_meta_suggestions_v2.json"

    def __init__(self):
        """Initialize V2 meta-learner."""
        # Create directories if needed
        self.CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def load_recent_corrections(self, hours: int = 24) -> List[Dict]:
        """
        Load VLM corrections from recent V2 sessions.

        Args:
            hours: How far back to look

        Returns:
            List of correction records
        """
        corrections = []
        cutoff_time = time.time() - (hours * 3600)

        if not self.CORRECTIONS_DIR.exists():
            return corrections

        # Load V2 VLM corrections
        for corrections_file in self.CORRECTIONS_DIR.glob("v2_corrections_*.json"):
            if corrections_file.stat().st_mtime < cutoff_time:
                continue

            try:
                with open(corrections_file, 'r') as f:
                    data = json.load(f)

                for c in data.get('corrections', []):
                    c['source'] = 'vlm_v2'
                    c['file'] = corrections_file.name
                    corrections.append(c)

            except Exception as e:
                print(f"Error loading {corrections_file}: {e}")

        return corrections

    def analyze_correction_patterns(self, corrections: List[Dict]) -> Dict:
        """
        Analyze patterns in V2 corrections.

        Returns statistics about:
        - Strategist: Mode selection errors
        - Tactician: Target selection errors
        - Executor: Mouse positioning errors
        """
        stats = {
            'total': len(corrections),
            'by_level': {
                'strategist': 0,
                'tactician': 0,
                'executor': 0,
                'unknown': 0
            },
            'strategist_issues': defaultdict(int),
            'tactician_issues': defaultdict(int),
            'executor_issues': defaultdict(int),
            'mode_errors': defaultdict(int),
            'target_errors': defaultdict(int),
            'mouse_positions': [],  # For clustering analysis
            'target_positions': [],  # For bias detection
            'click_out_of_bounds': 0,
        }

        for c in corrections:
            vlm_result = c.get('vlm_result', {})
            policy_output = c.get('policy_output', {})

            # Determine correction level from VLM result structure
            level = 'unknown'

            # Strategist corrections
            if 'current_mode_correct' in vlm_result:
                level = 'strategist'
                stats['by_level']['strategist'] += 1

                if not vlm_result.get('current_mode_correct', True):
                    current = c.get('mode', 'unknown')
                    recommended = vlm_result.get('recommended_mode', 'unknown')
                    stats['mode_errors'][f"{current}->{recommended}"] += 1

                    reason = vlm_result.get('mode_reason', '')
                    if reason:
                        stats['strategist_issues'][reason[:50]] += 1

            # Tactician corrections
            elif 'target_correct' in vlm_result:
                level = 'tactician'
                stats['by_level']['tactician'] += 1

                if not vlm_result.get('target_correct', True):
                    # Track target errors
                    current_class = c.get('target_class', 'unknown')
                    rec_target = vlm_result.get('recommended_target', {})
                    rec_class = rec_target.get('class', 'unknown')
                    stats['target_errors'][f"{current_class}->{rec_class}"] += 1

                    reason = vlm_result.get('reason', rec_target.get('reason', ''))
                    if reason:
                        stats['tactician_issues'][reason[:50]] += 1

                    # Track target positions for bias analysis
                    target_idx = c.get('target_idx', -1)
                    objects = c.get('objects', [])
                    if target_idx >= 0 and target_idx < len(objects):
                        obj = objects[target_idx]
                        if len(obj) >= 2:
                            stats['target_positions'].append((float(obj[0]), float(obj[1])))

            # Executor corrections
            elif 'movement_correct' in vlm_result:
                level = 'executor'
                stats['by_level']['executor'] += 1

                if not vlm_result.get('movement_correct', True):
                    quality = vlm_result.get('quality', 'unknown')
                    issue = vlm_result.get('issue', '')

                    if issue:
                        stats['executor_issues'][issue[:50]] += 1

                    # Track mouse positions for clustering analysis
                    action = policy_output.get('action', {})
                    mouse_x = action.get('mouse_x')
                    mouse_y = action.get('mouse_y')

                    if mouse_x is not None and mouse_y is not None:
                        stats['mouse_positions'].append((float(mouse_x), float(mouse_y)))

                        # Check out of bounds
                        if mouse_x < 0 or mouse_x > 1 or mouse_y < 0 or mouse_y > 1:
                            stats['click_out_of_bounds'] += 1

            else:
                stats['by_level']['unknown'] += 1

        return stats

    def detect_overfitting(self, stats: Dict) -> Dict[str, any]:
        """
        Detect overfitting patterns in executor behavior.

        Returns:
            Dictionary with overfitting diagnosis
        """
        diagnosis = {
            'executor_clustering': None,
            'target_clustering': None,
            'position_bias': None,
            'out_of_bounds_rate': 0.0,
            'critical_issues': []
        }

        # Analyze mouse position clustering
        mouse_positions = stats.get('mouse_positions', [])
        if len(mouse_positions) > 10:
            positions = np.array(mouse_positions)

            # Check for clustering in specific regions
            # Bottom-right corner overfitting (common issue)
            bottom_right = np.sum((positions[:, 0] > 0.7) & (positions[:, 1] > 0.7))
            br_ratio = bottom_right / len(positions)

            if br_ratio > 0.7:
                diagnosis['executor_clustering'] = {
                    'region': 'bottom-right',
                    'ratio': br_ratio,
                    'severity': 'critical' if br_ratio > 0.8 else 'high'
                }
                diagnosis['critical_issues'].append(
                    f"Executor clicks clustered in bottom-right ({br_ratio:.0%})"
                )

            # Check other quadrants
            quadrants = {
                'top-left': np.sum((positions[:, 0] < 0.3) & (positions[:, 1] < 0.3)),
                'top-right': np.sum((positions[:, 0] > 0.7) & (positions[:, 1] < 0.3)),
                'bottom-left': np.sum((positions[:, 0] < 0.3) & (positions[:, 1] > 0.7)),
            }

            for quad, count in quadrants.items():
                ratio = count / len(positions)
                if ratio > 0.6:
                    diagnosis['executor_clustering'] = {
                        'region': quad,
                        'ratio': ratio,
                        'severity': 'high'
                    }
                    diagnosis['critical_issues'].append(
                        f"Executor clicks clustered in {quad} ({ratio:.0%})"
                    )

            # Calculate spread (low spread = overfitting)
            std_x = np.std(positions[:, 0])
            std_y = np.std(positions[:, 1])

            if std_x < 0.15 or std_y < 0.15:
                diagnosis['position_bias'] = {
                    'std_x': std_x,
                    'std_y': std_y,
                    'issue': 'Low position variance suggests overfitting to specific locations'
                }

        # Analyze target position clustering
        target_positions = stats.get('target_positions', [])
        if len(target_positions) > 10:
            targets = np.array(target_positions)

            # Check if targets are clustered in one area
            std_x = np.std(targets[:, 0])
            std_y = np.std(targets[:, 1])

            if std_x < 0.2 or std_y < 0.2:
                diagnosis['target_clustering'] = {
                    'std_x': std_x,
                    'std_y': std_y,
                    'issue': 'Targets clustered in limited area - training data may be from one map location'
                }
                diagnosis['critical_issues'].append(
                    "Target positions show low variance - possible training data bias"
                )

        # Out of bounds rate
        total_clicks = len(mouse_positions)
        if total_clicks > 0:
            oob_rate = stats.get('click_out_of_bounds', 0) / total_clicks
            diagnosis['out_of_bounds_rate'] = oob_rate

            if oob_rate > 0.1:
                diagnosis['critical_issues'].append(
                    f"Clicks outside screen bounds in {oob_rate:.0%} of corrections"
                )

        return diagnosis

    def generate_suggestions(self, stats: Dict, overfitting: Dict) -> Dict:
        """
        Generate actionable suggestions for fixing issues.

        Args:
            stats: Correction pattern statistics
            overfitting: Overfitting diagnosis

        Returns:
            Suggestions dictionary
        """
        suggestions = {
            'strategist': [],
            'tactician': [],
            'executor': [],
            'training_data': [],
            'critical_fixes': []
        }

        total = stats['total']
        if total == 0:
            return suggestions

        # Strategist analysis
        strategist_count = stats['by_level']['strategist']
        if strategist_count > 0:
            error_rate = strategist_count / total

            if error_rate > 0.2:
                suggestions['strategist'].append({
                    'issue': f"Mode selection errors in {error_rate:.0%} of corrections",
                    'severity': 'high' if error_rate > 0.3 else 'medium',
                    'fix': 'Review strategist training data for mode transitions',
                    'details': f"Common errors: {dict(list(stats['mode_errors'].items())[:3])}"
                })

        # Tactician analysis
        tactician_count = stats['by_level']['tactician']
        if tactician_count > 0:
            error_rate = tactician_count / total

            if error_rate > 0.15:
                suggestions['tactician'].append({
                    'issue': f"Target selection errors in {error_rate:.0%} of corrections",
                    'severity': 'high' if error_rate > 0.25 else 'medium',
                    'fix': 'Review target prioritization logic or add more diverse target examples',
                    'details': f"Common errors: {dict(list(stats['target_errors'].items())[:3])}"
                })

            # Target clustering
            if overfitting.get('target_clustering'):
                tc = overfitting['target_clustering']
                suggestions['training_data'].append({
                    'issue': 'Target positions show low variance',
                    'severity': 'high',
                    'fix': 'Record training data from different map locations OR add position augmentation',
                    'details': f"std_x={tc['std_x']:.3f}, std_y={tc['std_y']:.3f}"
                })
                suggestions['critical_fixes'].append(
                    'Training data may be from one map location - record from multiple areas'
                )

        # Executor analysis
        executor_count = stats['by_level']['executor']
        if executor_count > 0:
            error_rate = executor_count / total

            # Clustering issues
            if overfitting.get('executor_clustering'):
                ec = overfitting['executor_clustering']
                suggestions['executor'].append({
                    'issue': f"Mouse clicks clustered in {ec['region']} ({ec['ratio']:.0%})",
                    'severity': ec['severity'],
                    'fix': 'CRITICAL: Executor is overfitting to one screen region',
                    'details': 'Possible causes: Training data all from same map area, model capacity too small, or insufficient data augmentation'
                })
                suggestions['critical_fixes'].append(
                    f"Executor overfitting to {ec['region']} - add diverse position training data"
                )

            # Position bias
            if overfitting.get('position_bias'):
                pb = overfitting['position_bias']
                suggestions['executor'].append({
                    'issue': 'Low position variance in mouse movements',
                    'severity': 'medium',
                    'fix': 'Add position noise augmentation during training',
                    'details': f"std_x={pb['std_x']:.3f}, std_y={pb['std_y']:.3f}"
                })

            # Out of bounds
            oob_rate = overfitting.get('out_of_bounds_rate', 0)
            if oob_rate > 0.1:
                suggestions['executor'].append({
                    'issue': f"Clicks outside screen bounds ({oob_rate:.0%})",
                    'severity': 'high',
                    'fix': 'Add coordinate clamping OR fix screen resolution normalization',
                    'details': 'Model predicting invalid positions'
                })
                suggestions['critical_fixes'].append(
                    'Executor clicking outside screen - check coordinate normalization'
                )

        return suggestions

    def print_summary(self, stats: Dict, overfitting: Dict, suggestions: Dict):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print("  V2 META-LEARNING ANALYSIS")
        print("="*60)
        print(f"\nAnalyzed {stats['total']} VLM corrections from last session\n")

        # Strategist
        strategist_count = stats['by_level']['strategist']
        total = max(stats['total'], 1)
        strategist_rate = strategist_count / total

        print("STRATEGIST Issues:")
        if strategist_rate < 0.1:
            print(f"  [OK] Mode selection {(1-strategist_rate):.0%} accurate")
        else:
            print(f"  [WARN] Mode selection errors: {strategist_count}/{total} corrections ({strategist_rate:.0%})")
            for error, count in list(stats['mode_errors'].items())[:3]:
                print(f"    - {error}: {count} times")

        # Tactician
        print("\nTACTICIAN Issues:")
        tactician_count = stats['by_level']['tactician']
        tactician_rate = tactician_count / total

        if tactician_rate < 0.1:
            print(f"  [OK] Target selection {(1-tactician_rate):.0%} accurate")
        else:
            print(f"  [WARN] Target selection errors: {tactician_count}/{total} corrections ({tactician_rate:.0%})")
            for error, count in list(stats['target_errors'].items())[:3]:
                print(f"    - {error}: {count} times")

        # Executor
        print("\nEXECUTOR Issues:")
        executor_count = stats['by_level']['executor']

        if overfitting.get('executor_clustering'):
            ec = overfitting['executor_clustering']
            print(f"  [CRITICAL] Mouse clicks clustered in {ec['region']} ({ec['ratio']:.0%})")

        if overfitting.get('out_of_bounds_rate', 0) > 0.1:
            oob = overfitting['out_of_bounds_rate']
            print(f"  [CRITICAL] Clicks outside screen bounds in {oob:.0%} of corrections")

        if overfitting.get('position_bias'):
            pb = overfitting['position_bias']
            print(f"  [WARN] Low position variance (std_x={pb['std_x']:.3f}, std_y={pb['std_y']:.3f})")

        if not overfitting.get('executor_clustering') and overfitting.get('out_of_bounds_rate', 0) <= 0.1:
            print(f"  [OK] Movement quality acceptable")

        # Critical fixes
        if suggestions['critical_fixes']:
            print("\n[!] CRITICAL FIXES NEEDED:")
            for fix in suggestions['critical_fixes']:
                print(f"  - {fix}")

        # Training data suggestions
        if suggestions['training_data']:
            print("\n[DATA] Training Data Recommendations:")
            for rec in suggestions['training_data']:
                print(f"  - {rec['fix']}")
                print(f"    Reason: {rec['issue']}")

        print(f"\n[OK] Suggestions saved to: {self.SUGGESTIONS_FILE}")
        print("="*60 + "\n")

    def analyze_session(self, hours: int = 24) -> Optional[Dict]:
        """
        Main entry point - analyze recent V2 corrections.

        Args:
            hours: How far back to analyze

        Returns:
            Analysis results dict
        """
        print(f"\n[V2-META] Loading corrections from last {hours} hours...")
        corrections = self.load_recent_corrections(hours)

        if not corrections:
            print(f"[V2-META] No corrections found to analyze")
            return None

        print(f"[V2-META] Found {len(corrections)} corrections")

        # Analyze patterns
        stats = self.analyze_correction_patterns(corrections)

        # Detect overfitting
        overfitting = self.detect_overfitting(stats)

        # Generate suggestions
        suggestions = self.generate_suggestions(stats, overfitting)

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_corrections': len(corrections),
            'statistics': {
                'by_level': stats['by_level'],
                'mode_errors': dict(stats['mode_errors']),
                'target_errors': dict(stats['target_errors']),
                'out_of_bounds': stats['click_out_of_bounds']
            },
            'overfitting': {
                'executor_clustering': overfitting.get('executor_clustering'),
                'target_clustering': overfitting.get('target_clustering'),
                'position_bias': overfitting.get('position_bias'),
                'out_of_bounds_rate': overfitting.get('out_of_bounds_rate')
            },
            'suggestions': suggestions
        }

        # Convert numpy types to JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        results = make_serializable(results)

        with open(self.SUGGESTIONS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        self.print_summary(stats, overfitting, suggestions)

        return results


def main():
    """Run V2 meta-analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="V2 Meta-Learner - Analyze hierarchical policy corrections")
    parser.add_argument('--hours', type=int, default=24,
                       help='How many hours back to analyze (default: 24)')
    args = parser.parse_args()

    learner = MetaLearnerV2()
    learner.analyze_session(hours=args.hours)


if __name__ == "__main__":
    main()
