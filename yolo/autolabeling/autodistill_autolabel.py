"""Auto-label using Autodistill (Roboflow's official framework).

Uses Grounding DINO (better than CLIP) for zero-shot detection.

Install:
    pip install autodistill autodistill-grounding-dino

Usage:
    python autodistill_autolabel.py
"""
from pathlib import Path
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO


def autolabel_with_autodistill(
    input_dir: str,
    output_dir: str,
    class_prompts: dict,
    confidence: float = 0.35
):
    """
    Auto-label using Grounding DINO via Autodistill.

    Args:
        input_dir: Directory with images
        output_dir: Output directory
        class_prompts: Dict mapping class names to text prompts
                      e.g., {"enemy": "enemy spaceship", "player": "player ship"}
        confidence: Detection confidence threshold
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print("="*60)
    print("  AUTODISTILL AUTO-LABELING")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Classes: {list(class_prompts.keys())}")
    print(f"Confidence: {confidence}")
    print("="*60)

    # Create ontology (mapping from prompts to class names)
    ontology = CaptionOntology(class_prompts)

    # Load Grounding DINO (downloads ~300MB model first time)
    print("\nLoading Grounding DINO...")
    base_model = GroundingDINO(ontology=ontology)

    # Label images (creates YOLO format labels automatically)
    print("\nLabeling images...")
    base_model.label(
        input_folder=str(input_path),
        extension=".jpg",  # or .png
        output_folder=str(output_path),
    )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Output:")
    print(f"  {output_path}/images/")
    print(f"  {output_path}/labels/")
    print(f"\nNext: Review in Roboflow, then merge with dataset")


def main():
    """Example usage for DarkOrbit."""

    # Define your classes with text prompts
    # Grounding DINO works better with descriptive prompts
    class_prompts = {
        "enemy": "enemy spaceship",
        "player": "player spaceship",
        "laser": "laser beam",
        "explosion": "explosion effect",
        "npc": "NPC ship",
        "cargo": "cargo box",
        "portal": "portal gate",
        # Add more classes...
    }

    autolabel_with_autodistill(
        input_dir='F:/dev/bot/yolo/training_screenshots',
        output_dir='F:/dev/bot/yolo/autolabeled_autodistill',
        class_prompts=class_prompts,
        confidence=0.35
    )


if __name__ == '__main__':
    main()
