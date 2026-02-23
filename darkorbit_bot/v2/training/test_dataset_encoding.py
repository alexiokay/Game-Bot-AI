
import sys
import numpy as np
from pathlib import Path

# Mock config availability
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from darkorbit_bot.v2.training.train_tactician import TacticianDataset
from darkorbit_bot.v2.config import ENEMY_CLASSES, PLAYER_CLASSES, LOOT_CLASSES

def test_dataset_encoding():
    print("Testing TacticianDataset encoding...")
    
    # Create a dummy dataset instance (don't load actual data)
    dataset = TacticianDataset.__new__(TacticianDataset)
    dataset.object_dim = 20
    dataset.max_objects = 16
    
    # Mock input data
    tracked_objects = [
        {'x': 0.5, 'y': 0.5, 'class_name': 'Devo'},        # Enemy
        {'x': 0.5, 'y': 0.5, 'class_name': 'Player'},      # Player
        {'x': 0.5, 'y': 0.5, 'class_name': 'Asteroid'},    # Other
        {'x': 0.5, 'y': 0.5, 'class_name': 'BonusBox'}     # Loot
    ]
    
    objects, mask = dataset._objects_to_features(tracked_objects)
    
    # Check Enemy
    print(f"Object 0 (Devo): Enemy={objects[0, 16]}, Loot={objects[0, 17]}, Player={objects[0, 18]}, Other={objects[0, 19]}")
    assert objects[0, 16] == 1.0, "Devo should be enemy"
    assert objects[0, 18] == 0.0
    
    # Check Player
    print(f"Object 1 (Player): Enemy={objects[1, 16]}, Loot={objects[1, 17]}, Player={objects[1, 18]}, Other={objects[1, 19]}")
    assert objects[1, 16] == 0.0
    assert objects[1, 18] == 1.0, "Player should be is_player"
    
    # Check Other
    print(f"Object 2 (Asteroid): Enemy={objects[2, 16]}, Loot={objects[2, 17]}, Player={objects[2, 18]}, Other={objects[2, 19]}")
    assert objects[2, 19] == 1.0, "Asteroid should be is_other"
    
    print("\n✅ Dataset encoding verification PASSED")

if __name__ == "__main__":
    try:
        test_dataset_encoding()
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
