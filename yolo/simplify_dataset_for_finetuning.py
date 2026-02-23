"""Simplify v6 dataset to match pretrained model's 7 classes for fine-tuning.

Pretrained model classes:
  0: BonusBox
  1: Explosion
  2: GUI
  3: Lordakia
  4: Struener
  5: player_attack
  6: selection_ring
"""
import shutil
from pathlib import Path
import yaml

# V6 dataset classes (72 classes)
V6_CLASSES = ['AllyPlayer', 'AmmoDisplay', 'BonusBox', 'BoostersWindow', 'Button', 'CargoBox', 'ChatBox',
              'ChatWindow', 'Drone_havoc', 'Drone_iris', 'Drone_vandal', 'EnemyPlayer', 'Enemy_npc_boss_devolarium',
              'Enemy_npc_boss_kristalin', 'Enemy_npc_boss_kristalion', 'Enemy_npc_devolarium', 'Enemy_npc_elite_saimon',
              'Enemy_npc_elite_sylox', 'Enemy_npc_kristalin', 'Enemy_npc_kristalion', 'Enemy_npc_mordon', 'Enemy_npc_saimon',
              'Enemy_npc_sibelon', 'Enemy_npc_sylox', 'Enemy_npc_uber_kristallion', 'Enemy_target', 'EquipmentWindow',
              'Explosion', 'GroupWindow', 'Hotbar', 'HotbarWindow', 'Lasers', 'LogWindow', 'MenuBar', 'Minimap',
              'MissionObjective', 'MissionsWindow', 'Planet', 'Player', 'PlayerShip', 'Portal', 'Portal_btn', 'Saimon',
              'ShipWindow', 'Ship_bigboy', 'Ship_goliath', 'Ship_lenov', 'Ship_nostromo', 'Ship_phoenix', 'Ship_piranha',
              'Ship_senitel', 'Ship_solace', 'Ship_venegance', 'Ship_vengeance', 'Ship_venom', 'Sidebar_menu', 'Space_station',
              'Streuner', 'UserInfo', 'UserWindow', 'damage_numbers', 'drone_hercules', 'enemy_npc_boss_streuner',
              'enemy_npc_streuneR', 'enemy_npc_streuner', 'enemy_npc_uber_kristallon', 'lasers_sab', 'mine',
              'module_repair', 'ore', 'ore_palladium', 'sidebar_menu']

# Simplified 7-class mapping
SIMPLIFIED_CLASSES = ['BonusBox', 'Explosion', 'GUI', 'Lordakia', 'Struener', 'player_attack', 'selection_ring']

# Map v6 classes to simplified classes
CLASS_MAPPING = {
    # Direct matches
    'BonusBox': 'BonusBox',
    'Explosion': 'Explosion',

    # All Streuner variants -> Struener
    'Streuner': 'Struener',
    'enemy_npc_streuner': 'Struener',
    'enemy_npc_streuneR': 'Struener',
    'enemy_npc_boss_streuner': 'Struener',

    # All enemy NPCs -> Lordakia (generic alien enemy)
    'Enemy_npc_kristalin': 'Lordakia',
    'Enemy_npc_kristalion': 'Lordakia',
    'Enemy_npc_boss_kristalin': 'Lordakia',
    'Enemy_npc_boss_kristalion': 'Lordakia',
    'Enemy_npc_mordon': 'Lordakia',
    'Enemy_npc_saimon': 'Lordakia',
    'Enemy_npc_sibelon': 'Lordakia',
    'Enemy_npc_elite_saimon': 'Lordakia',
    'Enemy_npc_elite_sylox': 'Lordakia',
    'Enemy_npc_sylox': 'Lordakia',
    'Enemy_npc_devolarium': 'Lordakia',
    'Enemy_npc_boss_devolarium': 'Lordakia',
    'Enemy_npc_uber_kristallion': 'Lordakia',
    'enemy_npc_uber_kristallon': 'Lordakia',
    'Saimon': 'Lordakia',

    # All GUI/window elements -> GUI
    'ChatBox': 'GUI',
    'ChatWindow': 'GUI',
    'EquipmentWindow': 'GUI',
    'GroupWindow': 'GUI',
    'HotbarWindow': 'GUI',
    'LogWindow': 'GUI',
    'MenuBar': 'GUI',
    'MissionsWindow': 'GUI',
    'ShipWindow': 'GUI',
    'UserWindow': 'GUI',
    'BoostersWindow': 'GUI',
    'AmmoDisplay': 'GUI',
    'Hotbar': 'GUI',
    'Minimap': 'GUI',
    'Sidebar_menu': 'GUI',
    'sidebar_menu': 'GUI',
    'UserInfo': 'GUI',
    'Button': 'GUI',
    'MissionObjective': 'GUI',

    # Player attacks (lasers) -> player_attack
    'Lasers': 'player_attack',
    'lasers_sab': 'player_attack',

    # Enemy target ring -> selection_ring
    'Enemy_target': 'selection_ring',

    # Drop other classes that don't map well (remove from dataset)
    # CargoBox, Drone_*, EnemyPlayer, AllyPlayer, Player, PlayerShip, Portal,
    # Portal_btn, Ship_*, Space_station, Planet, damage_numbers, drone_hercules,
    # mine, module_repair, ore, ore_palladium
}

def main():
    print("=" * 70)
    print("SIMPLIFY DATASET FOR FINE-TUNING")
    print("=" * 70)

    print(f"\nOriginal v6 classes: {len(V6_CLASSES)}")
    print(f"Simplified classes: {len(SIMPLIFIED_CLASSES)}")
    print(f"Mapped v6 classes: {len(CLASS_MAPPING)}")
    print(f"Dropped classes: {len(V6_CLASSES) - len(CLASS_MAPPING)}")

    print(f"\n\nClass mapping:")
    for simplified_cls in SIMPLIFIED_CLASSES:
        v6_classes = [k for k, v in CLASS_MAPPING.items() if v == simplified_cls]
        print(f"\n  {simplified_cls} ({len(v6_classes)} v6 classes):")
        for v6_cls in sorted(v6_classes):
            print(f"    - {v6_cls}")

    dropped = [cls for cls in V6_CLASSES if cls not in CLASS_MAPPING]
    print(f"\n\nDropped classes ({len(dropped)}):")
    for cls in sorted(dropped):
        print(f"  - {cls}")

    # Create simplified dataset
    v6_dir = Path('F:/dev/bot/yolo/datasets/darkorbit_v6')
    simple_dir = Path('F:/dev/bot/yolo/datasets/darkorbit_v7_simple')

    print(f"\n\n" + "=" * 70)
    print("CREATING SIMPLIFIED DATASET")
    print("=" * 70)
    print(f"Source: {v6_dir}")
    print(f"Output: {simple_dir}")

    simple_dir.mkdir(exist_ok=True)

    # Create data.yaml
    data_yaml = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': len(SIMPLIFIED_CLASSES),
        'names': SIMPLIFIED_CLASSES,
    }

    yaml_path = simple_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n[OK] Created data.yaml with {len(SIMPLIFIED_CLASSES)} classes")

    # Process each split
    total_images = 0
    total_labels = 0
    total_boxes_kept = 0
    total_boxes_dropped = 0

    for split in ['train', 'valid', 'test']:
        split_images = 0
        split_labels = 0
        split_boxes_kept = 0
        split_boxes_dropped = 0

        # Create directories
        (simple_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (simple_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        v6_images = v6_dir / split / 'images'
        v6_labels = v6_dir / split / 'labels'

        if not v6_images.exists():
            print(f"\n[SKIP] {split.capitalize()}: source folder not found, skipping")
            continue

        # Copy images and remap labels
        for img_path in v6_images.glob('*.*'):
            label_path = v6_labels / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Process label file
            new_lines = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_idx = int(parts[0])
                        old_class_name = V6_CLASSES[old_class_idx]

                        # Check if this class should be kept
                        if old_class_name in CLASS_MAPPING:
                            new_class_name = CLASS_MAPPING[old_class_name]
                            new_class_idx = SIMPLIFIED_CLASSES.index(new_class_name)

                            # Rebuild line with new class index
                            new_line = f"{new_class_idx} {' '.join(parts[1:])}\n"
                            new_lines.append(new_line)
                            split_boxes_kept += 1
                        else:
                            split_boxes_dropped += 1

            # Only copy image if it has at least one valid label
            if new_lines:
                # Copy image
                shutil.copy2(img_path, simple_dir / split / 'images' / img_path.name)
                split_images += 1

                # Write new label file
                with open(simple_dir / split / 'labels' / f"{img_path.stem}.txt", 'w') as f:
                    f.writelines(new_lines)
                split_labels += 1

        total_images += split_images
        total_labels += split_labels
        total_boxes_kept += split_boxes_kept
        total_boxes_dropped += split_boxes_dropped

        print(f"\n[OK] {split.capitalize():5s}: {split_images:3d} images, {split_labels:3d} labels")
        print(f"         Boxes kept: {split_boxes_kept:4d}, dropped: {split_boxes_dropped:4d}")

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Total boxes kept: {total_boxes_kept}")
    print(f"Total boxes dropped: {total_boxes_dropped}")
    print(f"Keep rate: {total_boxes_kept / (total_boxes_kept + total_boxes_dropped) * 100:.1f}%")

    print(f"\n[OK] Simplified dataset ready at: {simple_dir}")
    print(f"\nNow you can fine-tune the pretrained model with:")
    print(f"  model = YOLO('C:/Users/alexispace/Downloads/best (3).pt')")
    print(f"  model.train(data='{simple_dir}/data.yaml', epochs=100, lr0=0.001, ...)")

if __name__ == '__main__':
    main()
