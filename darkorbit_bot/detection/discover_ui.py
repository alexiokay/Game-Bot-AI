"""
DarkOrbit Bot - UI Element Discovery
First step: Analyze ONE screenshot to discover and name all UI elements.
Then use consistent naming for all labeling.

Usage:
    python discover_ui.py
"""

import os
import json
from pathlib import Path


def discover_ui_elements(image_path: str, api_key: str) -> dict:
    """
    Ask Gemini to analyze the game UI and create a vocabulary of elements.
    """
    from google import genai
    import PIL.Image
    
    client = genai.Client(api_key=api_key)
    img = PIL.Image.open(image_path)
    
    prompt = """You are analyzing a screenshot from a space game to create a complete UI element catalog.

Your task: Identify EVERY distinct UI element and game object visible in this screenshot.

For each element, provide:
1. A short, consistent name (snake_case, e.g. "ammo_bar_x1")
2. What it does / what it's for
3. Its approximate location (coordinates as percentages 0-100)

Be VERY detailed. Distinguish between:
- Individual buttons vs button groups
- Different types of bars (health vs shield vs experience)
- Different panels (group panel vs ship panel vs target panel)
- Clickable elements vs display-only elements
- Numbered slots (slot_1, slot_2, etc.)

Output as JSON:
{
    "ui_elements": [
        {
            "name": "minimap_panel",
            "description": "Navigation minimap showing nearby objects",
            "category": "navigation",
            "clickable": true,
            "x_min": 10, "y_min": 5, "x_max": 30, "y_max": 35
        },
        {
            "name": "ammo_slot_x1",
            "description": "First ammo type selector (x1 damage)",
            "category": "combat",
            "clickable": true,
            "hotkey": "1",
            "x_min": 42, "y_min": 85, "x_max": 48, "y_max": 95
        }
    ],
    "game_objects": [
        {
            "name": "enemy_ship",
            "description": "Enemy NPC spaceship",
            "category": "combat",
            "x_min": 60, "y_min": 20, "x_max": 65, "y_max": 25
        }
    ]
}

Be thorough! List EVERY visible element:
- All hotbar slots (numbered)
- All ammo types you can see
- Health bars, shield bars (distinguish player vs group members)
- Chat window
- Minimap
- Any buttons
- Skill icons
- Group member entries
- Ship status panel
- Any other UI elements

Output ONLY the JSON."""

    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[prompt, img]
    )
    
    try:
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        
        return json.loads(text.strip())
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Response:\n{response.text}")
        return {}


def save_ui_vocabulary(ui_data: dict, output_path: str):
    """Save the discovered UI vocabulary"""
    with open(output_path, 'w') as f:
        json.dump(ui_data, f, indent=2)
    print(f"\n💾 Saved to: {output_path}")


def display_results(ui_data: dict):
    """Display discovered elements"""
    print("\n" + "="*60)
    print("  📋 DISCOVERED UI ELEMENTS")
    print("="*60)
    
    if "ui_elements" in ui_data:
        print(f"\n🖼️  UI Elements ({len(ui_data['ui_elements'])}):")
        for elem in ui_data['ui_elements']:
            clickable = "🖱️" if elem.get('clickable') else "  "
            hotkey = f"[{elem.get('hotkey')}]" if elem.get('hotkey') else ""
            print(f"  {clickable} {elem['name']:30} {hotkey}")
            print(f"       {elem.get('description', '')[:50]}")
    
    if "game_objects" in ui_data:
        print(f"\n🎮 Game Objects ({len(ui_data['game_objects'])}):")
        for obj in ui_data['game_objects']:
            print(f"     {obj['name']:30}")
            print(f"       {obj.get('description', '')[:50]}")


def generate_class_mapping(ui_data: dict) -> dict:
    """Generate class_map for YOLO training"""
    class_map = {}
    idx = 0
    
    for elem in ui_data.get('ui_elements', []):
        class_map[elem['name']] = idx
        idx += 1
    
    for obj in ui_data.get('game_objects', []):
        class_map[obj['name']] = idx
        idx += 1
    
    return class_map


def main():
    print("\n" + "="*60)
    print("  🔍 UI ELEMENT DISCOVERY")
    print("  Step 1: Analyze game UI to create vocabulary")
    print("="*60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("\nEnter Gemini API Key: ").strip()
        if not api_key:
            return
    
    # Find a good sample image
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    images_dir = data_dir / "yolo_dataset" / "images" / "train"
    
    if not images_dir.exists():
        print(f"❌ No images found: {images_dir}")
        return
    
    images = list(images_dir.glob("*.png"))
    
    print(f"\n📸 Found {len(images)} screenshots")
    print("\nOptions:")
    print("  1. Use first screenshot")
    print("  2. Enter custom path")
    
    choice = input("\nChoice [1/2]: ").strip() or "1"
    
    if choice == "2":
        image_path = input("Image path: ").strip()
    else:
        image_path = str(images[0])
    
    print(f"\n🔍 Analyzing: {image_path}")
    print("   This may take a moment...")
    
    ui_data = discover_ui_elements(image_path, api_key)
    
    if not ui_data:
        print("❌ Failed to analyze image")
        return
    
    display_results(ui_data)
    
    # Save vocabulary
    vocab_path = data_dir / "ui_vocabulary.json"
    save_ui_vocabulary(ui_data, str(vocab_path))
    
    # Generate class mapping
    class_map = generate_class_mapping(ui_data)
    class_map_path = data_dir / "class_mapping.json"
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f, indent=2)
    print(f"💾 Class mapping saved to: {class_map_path}")
    
    print("\n" + "="*60)
    print("  ✅ NEXT STEPS")
    print("="*60)
    print(f"""
1. Review the vocabulary: {vocab_path}
2. Edit names if needed
3. Run labeling with this vocabulary:
   python detection/gemini_labeler.py --vocab {vocab_path}
""")


if __name__ == "__main__":
    main()
