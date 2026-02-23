"""Auto-label with SAM + Grounding DINO (best combination).

Grounding DINO detects objects, SAM creates precise masks.

Install:
    pip install autodistill autodistill-grounding-dino autodistill-grounded-sam

Usage:
    python autodistill_sam_autolabel.py
"""
from pathlib import Path
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM


def autolabel_with_grounded_sam(
    input_dir: str,
    output_dir: str,
    class_prompts: dict,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25
):
    """
    Auto-label using Grounded SAM (Grounding DINO + SAM).

    This is what Roboflow uses internally!

    Args:
        input_dir: Directory with images
        output_dir: Output directory
        class_prompts: Dict mapping class names to text prompts
        box_threshold: Box confidence threshold
        text_threshold: Text matching threshold
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print("="*60)
    print("  GROUNDED SAM AUTO-LABELING")
    print("  (Grounding DINO + SAM)")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Classes: {list(class_prompts.keys())}")
    print(f"Box threshold: {box_threshold}")
    print(f"Text threshold: {text_threshold}")
    print("="*60)

    # Create ontology
    ontology = CaptionOntology(class_prompts)

    # Load Grounded SAM (combines Grounding DINO + SAM)
    print("\nLoading Grounded SAM...")
    print("(This downloads ~300MB Grounding DINO + ~300MB SAM first time)")
    base_model = GroundedSAM(
        ontology=ontology,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # Label images with instance segmentation
    print("\nLabeling images...")
    base_model.label(
        input_folder=str(input_path),
        extension=".jpg",
        output_folder=str(output_path),
    )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Output:")
    print(f"  {output_path}/images/")
    print(f"  {output_path}/labels/  (YOLO polygon format)")
    print(f"\nNext: Review in Roboflow, merge with dataset")


def main():
    """Example usage for DarkOrbit with detailed prompts."""

    # Better prompts = better detection
    class_prompts = {
        # Spaceships
        "enemy-ship": "red enemy spaceship",
        "player-ship": "blue player spaceship flying in space",
        "npc-ship": "yellow NPC ship",

        # Combat
        "laser": "laser beam projectile",
        "rocket": "rocket missile",
        "explosion": "explosion effect in space",

        # Items
        "cargo": "green cargo box",
        "bonus-box": "bonus box resource",

        # Environment
        "portal": "portal gate entrance",
        "station": "space station",

        # UI (might not work well, Grounding DINO is for objects)
        # "minimap": "minimap user interface",
        # "health-bar": "health bar",
    }

    autolabel_with_grounded_sam(
        input_dir='F:/dev/bot/yolo/training_screenshots',
        output_dir='F:/dev/bot/yolo/autolabeled_grounded_sam',
        class_prompts=class_prompts,
        box_threshold=0.30,  # Lower = more detections
        text_threshold=0.25
    )


if __name__ == '__main__':
    main()
