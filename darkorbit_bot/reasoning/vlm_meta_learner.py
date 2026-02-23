"""
VLM Meta-Learner - Self-improving VLM system

After each bot session, this analyzes:
1. What VLM corrections were generated
2. Which corrections led to good/bad outcomes
3. What patterns the VLM missed or got wrong
4. Suggestions for improving the VLM system prompt

The meta-learner uses a "thinking" LLM to reflect on VLM performance
and propose improvements to the system prompt.

Usage:
    # After a bot session:
    python -m darkorbit_bot.reasoning.vlm_meta_learner

    # Or programmatically:
    from darkorbit_bot.reasoning.vlm_meta_learner import MetaLearner
    learner = MetaLearner()
    suggestions = learner.analyze_session()
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests


class MetaLearner:
    """
    Analyzes VLM performance and suggests improvements.

    This is a "meta" layer - it watches how the VLM watches the bot,
    and figures out how to make the VLM better.
    """

    # Where VLM corrections are stored
    CORRECTIONS_DIR = Path(__file__).parent.parent / "data" / "vlm_corrections"

    # Where the system prompt lives (what we want to improve)
    SYSTEM_PROMPT_FILE = Path(__file__).parent.parent / "data" / "vlm_system_prompt.txt"

    # Where we save meta-analysis results
    META_ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "meta_analysis"

    # Where we save suggested prompt improvements
    PROMPT_SUGGESTIONS_FILE = Path(__file__).parent.parent / "data" / "vlm_prompt_suggestions.md"

    def __init__(self,
                 base_url: str = "http://localhost:1234",
                 model: str = "local-model"):
        """
        Initialize meta-learner.

        Args:
            base_url: LM Studio API URL
            model: Model to use for meta-analysis (can be different from VLM)
        """
        self.base_url = base_url
        self.model = model

        # Create directories
        self.META_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    def get_current_system_prompt(self) -> str:
        """Load current VLM system prompt."""
        if self.SYSTEM_PROMPT_FILE.exists():
            return self.SYSTEM_PROMPT_FILE.read_text(encoding='utf-8')
        return ""

    def load_recent_corrections(self, hours: int = 24) -> List[Dict]:
        """
        Load VLM corrections from recent sessions.

        Args:
            hours: How far back to look

        Returns:
            List of correction records with metadata
        """
        corrections = []
        cutoff_time = time.time() - (hours * 3600)

        if not self.CORRECTIONS_DIR.exists():
            return corrections

        # Load self-improve corrections
        for session_dir in self.CORRECTIONS_DIR.glob("session_*"):
            corrections_file = session_dir / "corrections.json"
            if not corrections_file.exists():
                continue

            # Check if recent enough
            if corrections_file.stat().st_mtime < cutoff_time:
                continue

            try:
                with open(corrections_file, 'r') as f:
                    data = json.load(f)

                for c in data.get('corrections', []):
                    c['source'] = 'self_improve'
                    c['session'] = session_dir.name
                    corrections.append(c)

            except Exception as e:
                print(f"Error loading {corrections_file}: {e}")

        # Load enhanced VLM corrections
        for enhanced_file in self.CORRECTIONS_DIR.glob("enhanced_*.json"):
            if enhanced_file.stat().st_mtime < cutoff_time:
                continue

            try:
                with open(enhanced_file, 'r') as f:
                    data = json.load(f)

                for c in data.get('corrections', []):
                    c['source'] = 'enhanced_vlm'
                    c['file'] = enhanced_file.name
                    corrections.append(c)

            except Exception as e:
                print(f"Error loading {enhanced_file}: {e}")

        # Load bad stop corrections (user feedback)
        for bad_stop_file in self.CORRECTIONS_DIR.glob("bad_stop_*.json"):
            if bad_stop_file.stat().st_mtime < cutoff_time:
                continue

            try:
                with open(bad_stop_file, 'r') as f:
                    data = json.load(f)

                for c in data.get('corrections', []):
                    c['source'] = 'bad_stop'
                    c['file'] = bad_stop_file.name
                    corrections.append(c)

            except Exception as e:
                print(f"Error loading {bad_stop_file}: {e}")

        return corrections

    def load_debug_data(self, hours: int = 24) -> List[Dict]:
        """
        Load debug JSON files that show what VLM saw and decided.

        These contain the full VLM input/output for each critique.
        """
        debug_data = []
        cutoff_time = time.time() - (hours * 3600)

        # Look in session debug directories
        for session_dir in self.CORRECTIONS_DIR.glob("session_*"):
            debug_dir = session_dir / "debug"
            if not debug_dir.exists():
                continue

            for debug_file in debug_dir.glob("critique_*.json"):
                if debug_file.stat().st_mtime < cutoff_time:
                    continue

                try:
                    with open(debug_file, 'r') as f:
                        data = json.load(f)
                    data['file'] = str(debug_file)
                    debug_data.append(data)
                except Exception:
                    pass

        return debug_data

    def analyze_correction_patterns(self, corrections: List[Dict]) -> Dict:
        """
        Analyze patterns in VLM corrections.

        Returns statistics about:
        - Quality distribution (good/bad/needs_improvement)
        - Common issues detected
        - Combat tactic distribution
        - Bad stop frequency (user disagreements)
        """
        stats = {
            'total': len(corrections),
            'by_quality': {'good': 0, 'bad': 0, 'needs_improvement': 0, 'unknown': 0},
            'by_source': {'self_improve': 0, 'enhanced_vlm': 0, 'bad_stop': 0},
            'by_level': {'strategic': 0, 'tactical': 0, 'execution': 0},
            'combat_tactics': {},
            'main_issues': {},
            'bad_stop_count': 0,
        }

        for c in corrections:
            # Quality distribution
            quality = c.get('quality', c.get('bot_quality', 'unknown'))
            if quality in stats['by_quality']:
                stats['by_quality'][quality] += 1
            else:
                stats['by_quality']['unknown'] += 1

            # Source distribution
            source = c.get('source', 'unknown')
            if source in stats['by_source']:
                stats['by_source'][source] += 1

            # Level distribution (enhanced VLM)
            level = c.get('level')
            if level and level in stats['by_level']:
                stats['by_level'][level] += 1

            # Combat tactics
            tactics = c.get('combat_tactics', {})
            if isinstance(tactics, dict):
                tactic = tactics.get('detected_tactic', 'unknown')
                stats['combat_tactics'][tactic] = stats['combat_tactics'].get(tactic, 0) + 1

            # Main issues
            issue = c.get('main_issue', '')
            if issue:
                # Normalize issue text
                issue_key = issue[:50].lower().strip()
                stats['main_issues'][issue_key] = stats['main_issues'].get(issue_key, 0) + 1

            # Bad stop count
            if source == 'bad_stop':
                stats['bad_stop_count'] += 1

        return stats

    def generate_meta_analysis(self,
                               corrections: List[Dict],
                               debug_data: List[Dict],
                               stats: Dict) -> Optional[Dict]:
        """
        Use LLM to analyze VLM performance and suggest improvements.

        This is the "thinking" step - we show the LLM:
        1. Current system prompt
        2. Sample of VLM corrections
        3. Statistics about VLM performance
        4. Bad stop data (where user disagreed)

        And ask it to suggest improvements.
        """
        current_prompt = self.get_current_system_prompt()

        # Sample corrections for analysis (don't send too many)
        sample_corrections = corrections[:20] if len(corrections) > 20 else corrections

        # Focus on bad stops (user disagreements) - these are most valuable
        bad_stops = [c for c in corrections if c.get('source') == 'bad_stop']

        # Build analysis prompt
        analysis_prompt = f"""You are a meta-learning system analyzing the performance of a VLM (Vision Language Model) that watches a game bot play DarkOrbit.

=== CURRENT VLM SYSTEM PROMPT ===
{current_prompt}

=== VLM PERFORMANCE STATISTICS ===
Total corrections generated: {stats['total']}
Quality distribution: {json.dumps(stats['by_quality'], indent=2)}
Source distribution: {json.dumps(stats['by_source'], indent=2)}
Level distribution: {json.dumps(stats['by_level'], indent=2)}
Combat tactics detected: {json.dumps(stats['combat_tactics'], indent=2)}
Bad stops (user disagreements): {stats['bad_stop_count']}

=== BAD STOPS (User said VLM/bot was WRONG) ===
{json.dumps(bad_stops[:10], indent=2, default=str) if bad_stops else "No bad stops recorded"}

=== SAMPLE VLM CORRECTIONS ===
{json.dumps(sample_corrections[:10], indent=2, default=str)}

=== YOUR ANALYSIS TASK ===

Analyze the VLM's performance and suggest improvements. Consider:

1. ACCURACY: Is the VLM correctly identifying good vs bad bot behavior?
   - Look at the quality distribution
   - Are there many "unknown" or inconsistent ratings?

2. COVERAGE: What is the VLM missing?
   - Are there game situations not covered in the system prompt?
   - Are there enemy types or tactics not mentioned?

3. BAD STOPS: Why did the user disagree with the bot/VLM?
   - What patterns appear in bad stop data?
   - Was the VLM giving bad advice that led to these?

4. SPECIFICITY: Is the system prompt too vague or too specific?
   - Should distances/timings be more/less specific?
   - Are the behavior definitions clear enough?

5. COMBAT TACTICS: Is the VLM correctly detecting and recommending tactics?
   - Look at the combat_tactics distribution
   - Are good tactics being rewarded, bad tactics penalized?

Reply with JSON:
{{
    "overall_assessment": "Brief summary of VLM performance (1-2 sentences)",
    "strengths": ["what the VLM is doing well", "..."],
    "weaknesses": ["what the VLM is doing poorly", "..."],
    "missed_patterns": ["game situations the VLM missed", "..."],
    "bad_stop_analysis": "Why users are pressing F2 - what's going wrong?",
    "prompt_improvements": [
        {{
            "section": "which section of the prompt to change",
            "current": "current text (brief)",
            "suggested": "suggested new text",
            "reasoning": "why this change would help"
        }}
    ],
    "new_sections_to_add": [
        {{
            "title": "section title",
            "content": "what to add",
            "reasoning": "why this is needed"
        }}
    ],
    "confidence": 0.0 to 1.0,
    "priority_changes": ["most important changes to make first"]
}}"""

        # Query LLM for analysis
        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert AI systems analyst specializing in self-improving machine learning systems. You analyze VLM performance and suggest concrete improvements."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.7  # Some creativity for suggestions
            }

            print("🧠 Running meta-analysis (this may take a moment)...")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON from response
            # Handle markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            analysis = json.loads(content.strip())
            return analysis

        except requests.exceptions.RequestException as e:
            print(f"❌ LLM request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response: {e}")
            print(f"   Raw response: {content[:500]}...")
            return None

    def save_analysis(self, analysis: Dict, stats: Dict) -> Path:
        """Save meta-analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meta_analysis_{timestamp}.json"
        filepath = self.META_ANALYSIS_DIR / filename

        data = {
            'timestamp': timestamp,
            'statistics': stats,
            'analysis': analysis,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"📊 Analysis saved to: {filepath}")
        return filepath

    def generate_suggestions_markdown(self, analysis: Dict) -> str:
        """Generate human-readable markdown with improvement suggestions."""
        md = f"""# VLM Improvement Suggestions
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Assessment
{analysis.get('overall_assessment', 'No assessment available')}

## Strengths
"""
        for s in analysis.get('strengths', []):
            md += f"- {s}\n"

        md += "\n## Weaknesses\n"
        for w in analysis.get('weaknesses', []):
            md += f"- {w}\n"

        md += "\n## Missed Patterns\n"
        for p in analysis.get('missed_patterns', []):
            md += f"- {p}\n"

        md += f"\n## Bad Stop Analysis\n{analysis.get('bad_stop_analysis', 'No bad stops analyzed')}\n"

        md += "\n## Prompt Improvements\n"
        for i, imp in enumerate(analysis.get('prompt_improvements', []), 1):
            md += f"""
### {i}. {imp.get('section', 'Unknown Section')}

**Current:** {imp.get('current', 'N/A')}

**Suggested:** {imp.get('suggested', 'N/A')}

**Reasoning:** {imp.get('reasoning', 'N/A')}
"""

        md += "\n## New Sections to Add\n"
        for section in analysis.get('new_sections_to_add', []):
            md += f"""
### {section.get('title', 'Untitled')}

{section.get('content', '')}

*Reasoning: {section.get('reasoning', 'N/A')}*
"""

        md += f"\n## Priority Changes\n"
        for i, change in enumerate(analysis.get('priority_changes', []), 1):
            md += f"{i}. {change}\n"

        md += f"\n---\n*Confidence: {analysis.get('confidence', 0):.0%}*\n"

        return md

    def load_existing_analysis(self) -> Optional[Dict]:
        """
        Load the most recent meta-analysis from file.

        Returns:
            Analysis dict if found, None otherwise
        """
        if not self.META_ANALYSIS_DIR.exists():
            return None

        # Find most recent analysis file
        analysis_files = sorted(self.META_ANALYSIS_DIR.glob("meta_analysis_*.json"), reverse=True)
        if not analysis_files:
            return None

        try:
            with open(analysis_files[0], 'r') as f:
                data = json.load(f)
            print(f"📂 Loaded existing analysis: {analysis_files[0].name}")
            return data.get('analysis')
        except Exception as e:
            print(f"❌ Failed to load analysis: {e}")
            return None

    def show_diff(self, old_text: str, new_text: str):
        """
        Show a colored diff between old and new text.
        Red = removed, Green = added
        """
        import difflib

        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = difflib.unified_diff(old_lines, new_lines,
                                     fromfile='current_prompt',
                                     tofile='new_prompt',
                                     lineterm='')

        # ANSI color codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        CYAN = '\033[96m'
        RESET = '\033[0m'

        print("\n" + "="*60)
        print("  PROPOSED CHANGES (diff)")
        print("="*60)
        print(f"  {RED}--- removed{RESET}  {GREEN}+++ added{RESET}")
        print("-"*60)

        lines_shown = 0
        max_lines = 50  # Limit output

        for line in diff:
            if lines_shown >= max_lines:
                print(f"\n... (truncated, {len(new_lines)} total lines)")
                break

            line = line.rstrip('\n')
            if line.startswith('---') or line.startswith('+++'):
                print(f"{CYAN}{line}{RESET}")
            elif line.startswith('-'):
                print(f"{RED}{line}{RESET}")
            elif line.startswith('+'):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith('@@'):
                print(f"{CYAN}{line}{RESET}")
            else:
                print(line)
            lines_shown += 1

        print("="*60)

    def apply_suggestions(self, analysis: Dict, auto_apply: bool = False) -> bool:
        """
        Apply suggested improvements to the system prompt.

        Args:
            analysis: Meta-analysis results
            auto_apply: If True, apply without confirmation

        Returns:
            True if changes were applied
        """
        if not analysis.get('prompt_improvements') and not analysis.get('new_sections_to_add'):
            print("No prompt improvements to apply")
            return False

        current_prompt = self.get_current_system_prompt()
        new_prompt = current_prompt

        # Track what changes will be made
        changes_summary = []

        # Apply improvements
        for imp in analysis.get('prompt_improvements', []):
            current_text = imp.get('current', '')
            suggested_text = imp.get('suggested', '')

            if current_text and suggested_text and current_text in new_prompt:
                new_prompt = new_prompt.replace(current_text, suggested_text)
                changes_summary.append(f"✏️ UPDATE: {imp.get('section', 'unknown section')}")
                changes_summary.append(f"   Reason: {imp.get('reasoning', 'N/A')[:60]}...")

        # Add new sections
        for section in analysis.get('new_sections_to_add', []):
            title = section.get('title', '')
            content = section.get('content', '')

            if title and content:
                new_section = f"\n\n{title.upper()}:\n{content}"
                new_prompt += new_section
                changes_summary.append(f"➕ ADD SECTION: {title}")
                changes_summary.append(f"   Reason: {section.get('reasoning', 'N/A')[:60]}...")

        if new_prompt == current_prompt:
            print("No changes could be applied (text not found in prompt)")
            return False

        if not auto_apply:
            # Show summary of changes
            print("\n" + "="*60)
            print("  CHANGES TO BE APPLIED")
            print("="*60)
            for line in changes_summary:
                print(line)

            # Show actual diff
            self.show_diff(current_prompt, new_prompt)

            response = input("\nApply these changes? [y/N]: ").strip().lower()
            if response != 'y':
                print("Changes not applied")
                return False

        # Backup current prompt
        backup_path = self.SYSTEM_PROMPT_FILE.with_suffix('.txt.backup')
        backup_path.write_text(current_prompt, encoding='utf-8')
        print(f"📦 Backed up current prompt to: {backup_path}")

        # Apply new prompt
        self.SYSTEM_PROMPT_FILE.write_text(new_prompt, encoding='utf-8')
        print(f"✅ Applied new prompt to: {self.SYSTEM_PROMPT_FILE}")

        # Clear cached prompt in SelfImprover
        try:
            from darkorbit_bot.reasoning.self_improver import SelfImprover
            SelfImprover.reload_system_prompt()
            print("🔄 Reloaded system prompt in SelfImprover")
        except Exception:
            pass

        return True

    def analyze_session(self,
                        hours: int = 24,
                        apply_changes: bool = False) -> Optional[Dict]:
        """
        Main entry point - analyze recent VLM performance and suggest improvements.

        Args:
            hours: How far back to analyze
            apply_changes: If True, apply suggested changes to system prompt

        Returns:
            Analysis results dict
        """
        print("\n" + "="*60)
        print("  🧠 VLM META-LEARNER")
        print("="*60)

        # Load data
        print(f"\n📂 Loading corrections from last {hours} hours...")
        corrections = self.load_recent_corrections(hours)
        debug_data = self.load_debug_data(hours)

        if not corrections:
            print("❌ No corrections found to analyze")
            return None

        print(f"   Found {len(corrections)} corrections, {len(debug_data)} debug records")

        # Analyze patterns
        print("\n📊 Analyzing patterns...")
        stats = self.analyze_correction_patterns(corrections)

        print(f"   Quality: good={stats['by_quality']['good']}, "
              f"bad={stats['by_quality']['bad']}, "
              f"needs_improvement={stats['by_quality']['needs_improvement']}")
        print(f"   Bad stops (user disagreements): {stats['bad_stop_count']}")

        # Generate meta-analysis
        print("\n🤔 Generating meta-analysis...")
        analysis = self.generate_meta_analysis(corrections, debug_data, stats)

        if not analysis:
            print("❌ Meta-analysis failed")
            return None

        # Save results
        self.save_analysis(analysis, stats)

        # Generate and save suggestions markdown
        suggestions_md = self.generate_suggestions_markdown(analysis)
        self.PROMPT_SUGGESTIONS_FILE.write_text(suggestions_md, encoding='utf-8')
        print(f"📝 Suggestions saved to: {self.PROMPT_SUGGESTIONS_FILE}")

        # Print summary
        print("\n" + "="*60)
        print("  ANALYSIS SUMMARY")
        print("="*60)
        print(f"\n{analysis.get('overall_assessment', 'No assessment')}")

        print("\n🎯 Priority changes:")
        for i, change in enumerate(analysis.get('priority_changes', [])[:3], 1):
            print(f"   {i}. {change}")

        print(f"\n💡 {len(analysis.get('prompt_improvements', []))} prompt improvements suggested")
        print(f"➕ {len(analysis.get('new_sections_to_add', []))} new sections suggested")
        print(f"🔮 Confidence: {analysis.get('confidence', 0):.0%}")

        # Apply changes if requested
        if apply_changes:
            print("\n" + "-"*60)
            self.apply_suggestions(analysis, auto_apply=False)

        return analysis


def main():
    """Run meta-analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="VLM Meta-Learner - Analyze and improve VLM performance")
    parser.add_argument('--hours', type=int, default=24,
                        help='How many hours back to analyze (default: 24)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply previously generated suggestions (no new analysis)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run new analysis (default if no other flags)')
    parser.add_argument('--url', type=str, default='http://localhost:1234',
                        help='LM Studio URL')
    args = parser.parse_args()

    learner = MetaLearner(base_url=args.url)

    # If --apply is used alone, just apply existing suggestions without re-analyzing
    if args.apply and not args.analyze:
        print("\n" + "="*60)
        print("  🧠 APPLYING EXISTING SUGGESTIONS")
        print("="*60)

        analysis = learner.load_existing_analysis()
        if not analysis:
            print("\n❌ No existing analysis found!")
            print(f"   Run without --apply first to generate analysis")
            return

        # Show summary of what will be applied
        print(f"\n📋 Analysis summary:")
        print(f"   {analysis.get('overall_assessment', 'No assessment')}")
        print(f"   {len(analysis.get('prompt_improvements', []))} prompt improvements")
        print(f"   {len(analysis.get('new_sections_to_add', []))} new sections")

        if learner.apply_suggestions(analysis, auto_apply=False):
            print("\n✅ Suggestions applied!")
        else:
            print("\n⚠️ No changes applied")
        return

    # Otherwise, run new analysis
    analysis = learner.analyze_session(hours=args.hours, apply_changes=False)

    if analysis:
        print("\n✅ Meta-analysis complete!")
        print(f"   View suggestions: {learner.PROMPT_SUGGESTIONS_FILE}")
        print(f"   To apply changes: python -m darkorbit_bot.reasoning.vlm_meta_learner --apply")
    else:
        print("\n❌ Meta-analysis failed - check LM Studio connection")


if __name__ == "__main__":
    main()
