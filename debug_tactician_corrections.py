import json

data = json.load(open('f:/dev/bot/darkorbit_bot/data/vlm_corrections_v2/v2_corrections_1769226324.json'))

print('=== TACTICIAN CORRECTIONS DEBUG ===\n')

tact_count = 0
for i, c in enumerate(data['corrections']):
    vlm = c.get('vlm_result', {})
    if 'target_correct' in vlm or 'recommended_target' in vlm:
        tact_count += 1
        print(f"Correction #{i}:")
        print(f"  target_correct: {vlm.get('target_correct')}")
        print(f"  recommended_target: {vlm.get('recommended_target')}")
        print(f"  object_track_ids: {c.get('object_track_ids', [])}")
        print(f"  objects length: {len(c.get('objects', []))}")
        print(f"  object_mask length: {len(c.get('object_mask', []))}")

        # Try to find the matching logic
        rec = vlm.get('recommended_target', {})
        rec_id = rec.get('id', -1)
        track_ids = c.get('object_track_ids', [])
        print(f"  recommended id: {rec_id}")
        print(f"  track_ids: {track_ids}")

        if rec_id >= 0 and track_ids:
            for idx, tid in enumerate(track_ids):
                if tid == rec_id:
                    print(f"  ✓ MATCH found at index {idx}")
                    break
            else:
                print(f"  ✗ NO MATCH - recommended id {rec_id} not in track_ids")
        print()

print(f"\nTotal tactician corrections: {tact_count}")
