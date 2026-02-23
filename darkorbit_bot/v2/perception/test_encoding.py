
import sys
import numpy as np
from pathlib import Path

# Mock config availability
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from darkorbit_bot.v2.perception.tracker import TrackedObject
from darkorbit_bot.v2.config import ENEMY_CLASSES, PLAYER_CLASSES, LOOT_CLASSES

def test_object_encoding():
    print("Testing TrackedObject encoding...")
    
    # CASE 1: Enemy
    enemy = TrackedObject(track_id=1, class_name="Devo", x=0.5, y=0.5, width=0.1, height=0.1, confidence=0.9)
    feats = enemy.to_feature_vector()
    print(f"Enemy 'Devo' -> is_enemy={feats[16]}, is_loot={feats[17]}, is_player={feats[18]}, is_other={feats[19]}")
    assert feats[16] == 1.0, "Devo should be enemy"
    assert feats[18] == 0.0, "Devo should not be player"
    
    # CASE 2: Player (The issue case)
    player = TrackedObject(track_id=2, class_name="Player", x=0.5, y=0.5, width=0.1, height=0.1, confidence=0.9)
    feats = player.to_feature_vector()
    print(f"Player 'Player' -> is_enemy={feats[16]}, is_loot={feats[17]}, is_player={feats[18]}, is_other={feats[19]}")
    assert feats[16] == 0.0, "Player should not be enemy (by default)"
    assert feats[18] == 1.0, "Player should be is_player"
    
    # CASE 3: PlayerShip (Alternative name)
    player2 = TrackedObject(track_id=3, class_name="player_ship", x=0.5, y=0.5, width=0.1, height=0.1, confidence=0.9)
    feats = player2.to_feature_vector()
    print(f"Player 'player_ship' -> is_enemy={feats[16]}, is_loot={feats[17]}, is_player={feats[18]}, is_other={feats[19]}")
    assert feats[18] == 1.0, "player_ship should be is_player"
    
    # CASE 4: Unknown
    unknown = TrackedObject(track_id=4, class_name="Asteroid", x=0.5, y=0.5, width=0.1, height=0.1, confidence=0.9)
    feats = unknown.to_feature_vector()
    print(f"Unknown 'Asteroid' -> is_enemy={feats[16]}, is_loot={feats[17]}, is_player={feats[18]}, is_other={feats[19]}")
    assert feats[19] == 1.0, "Asteroid should be is_other"
    
    print("\n✅ TrackedObject encoding verification PASSED")

if __name__ == "__main__":
    try:
        test_object_encoding()
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
