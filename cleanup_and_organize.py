"""
Data Cleanup and Organization Script

This script:
1. Shows what's taking up space
2. Identifies V1 vs V2 data
3. Offers to clean up unnecessary files
4. Reorganizes into clear V1/V2 structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def get_dir_size(path):
    """Get directory size in MB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total / (1024 * 1024)  # Convert to MB


def analyze_data_folders():
    """Analyze all data folders"""
    print("=" * 80)
    print("DATA FOLDER ANALYSIS")
    print("=" * 80)

    folders = [
        ("data/recordings", "V2 Shadow recordings (with screenshots - HUGE)"),
        ("data/checkpoints", "Empty checkpoints"),
        ("darkorbit_bot/data/recordings", "V1 recordings"),
        ("darkorbit_bot/data/recordings_v2", "V2 JSON sequences"),
        ("darkorbit_bot/data/vlm_corrections", "V1 VLM corrections"),
        ("darkorbit_bot/data/vlm_corrections_v2", "V2 VLM corrections"),
        ("darkorbit_bot/data/v2_corrections", "V2 corrections (alt)"),
        ("darkorbit_bot/data/checkpoints", "Old checkpoints"),
        ("darkorbit_bot/data/bootstrap", "Bootstrap data"),
        ("darkorbit_bot/data/grounded", "Grounded data"),
        ("darkorbit_bot/data/corrections", "General corrections"),
        ("darkorbit_bot/data/yolo_dataset", "YOLO training data"),
    ]

    total_size = 0
    results = []

    for folder, description in folders:
        if Path(folder).exists():
            size_mb = get_dir_size(folder)
            total_size += size_mb
            results.append((folder, description, size_mb))
        else:
            results.append((folder, description, 0))

    # Sort by size
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Folder':<45} {'Size':>10}  Description")
    print("-" * 80)
    for folder, desc, size in results:
        if size >= 1024:
            size_str = f"{size/1024:.2f} GB"
        else:
            size_str = f"{size:.0f} MB"

        exists = "✓" if Path(folder).exists() else "✗"
        print(f"{exists} {folder:<42} {size_str:>10}  {desc}")

    print("-" * 80)
    if total_size >= 1024:
        print(f"Total: {total_size/1024:.2f} GB")
    else:
        print(f"Total: {total_size:.0f} MB")

    return results


def identify_problems(results):
    """Identify problematic files"""
    print("\n" + "=" * 80)
    print("PROBLEMS DETECTED")
    print("=" * 80)

    problems = []

    for folder, desc, size in results:
        # Shadow recording with screenshots is too large
        if "shadow" in folder.lower() and size > 10000:  # > 10 GB
            problems.append({
                'folder': folder,
                'issue': f"Shadow recording contains screenshots ({size/1024:.1f} GB)",
                'recommendation': "Delete and re-record without save_full_demos=True",
                'severity': 'HIGH'
            })

        # Old checkpoints
        if "checkpoints" in folder and size > 100 and "data/checkpoints" not in folder:
            problems.append({
                'folder': folder,
                'issue': f"Old checkpoints ({size:.0f} MB)",
                'recommendation': "Archive or delete if outdated",
                'severity': 'LOW'
            })

    if not problems:
        print("\n✓ No major problems detected!")
        return problems

    for i, prob in enumerate(problems, 1):
        severity_color = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        print(f"\n{severity_color[prob['severity']]} Problem #{i} ({prob['severity']} severity)")
        print(f"   Folder: {prob['folder']}")
        print(f"   Issue: {prob['issue']}")
        print(f"   Recommendation: {prob['recommendation']}")

    return problems


def propose_reorganization():
    """Propose clean folder structure"""
    print("\n" + "=" * 80)
    print("PROPOSED REORGANIZATION")
    print("=" * 80)

    print("""
Recommended structure:

data/
├── v1/                          # All V1 bot data
│   ├── recordings/              # V1 recordings (from darkorbit_bot/data/recordings)
│   ├── vlm_corrections/         # V1 VLM corrections
│   └── checkpoints/             # V1 checkpoints
│
├── v2/                          # All V2 bot data
│   ├── recordings/              # V2 recordings (from darkorbit_bot/data/recordings_v2)
│   ├── shadow_recordings/       # V2 shadow recordings (NO screenshots)
│   ├── vlm_corrections/         # V2 VLM corrections
│   └── checkpoints/             # V2 checkpoints
│
├── yolo/                        # YOLO detection training
│   └── training_data/           # YOLO datasets
│
└── archive/                     # Old/unused data
    └── shadow_with_screenshots/ # 19GB shadow recording (archive)

This structure:
- Clearly separates V1 and V2
- Removes confusion about which folder to use
- Archives large unnecessary files
- Makes training commands obvious
    """)


def cleanup_options():
    """Present cleanup options"""
    print("\n" + "=" * 80)
    print("CLEANUP OPTIONS")
    print("=" * 80)

    options = [
        {
            'id': 1,
            'name': 'Delete 19GB shadow recording',
            'action': 'Delete data/recordings/shadow_recording_20260123_000916.pkl',
            'saves': '19 GB',
            'risk': 'MEDIUM - You can re-record without screenshots'
        },
        {
            'id': 2,
            'name': 'Archive 19GB shadow recording',
            'action': 'Move to data/archive/shadow_with_screenshots/',
            'saves': '0 GB (just moves it)',
            'risk': 'LOW - Keeps data but out of the way'
        },
        {
            'id': 3,
            'name': 'Reorganize to V1/V2 structure',
            'action': 'Move folders to data/v1/ and data/v2/',
            'saves': '0 GB (just reorganizes)',
            'risk': 'LOW - Makes structure cleaner'
        },
        {
            'id': 4,
            'name': 'Clean old checkpoints',
            'action': 'Delete darkorbit_bot/data/checkpoints/',
            'saves': '20 MB',
            'risk': 'LOW - Old checkpoints probably unused'
        },
        {
            'id': 5,
            'name': 'Do nothing (just analyze)',
            'action': 'No changes',
            'saves': '0 GB',
            'risk': 'NONE'
        }
    ]

    for opt in options:
        print(f"\n[{opt['id']}] {opt['name']}")
        print(f"    Action: {opt['action']}")
        print(f"    Saves: {opt['saves']}")
        print(f"    Risk: {opt['risk']}")

    return options


def execute_cleanup(option_id):
    """Execute cleanup option"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if option_id == 1:
        # Delete shadow recording
        file_path = Path("data/recordings/shadow_recording_20260123_000916.pkl")
        if file_path.exists():
            print(f"\n🗑️  Deleting {file_path}...")
            file_path.unlink()
            print("✓ Deleted!")
        else:
            print("✗ File not found")

    elif option_id == 2:
        # Archive shadow recording
        src = Path("data/recordings/shadow_recording_20260123_000916.pkl")
        dst_dir = Path("data/archive/shadow_with_screenshots")
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"shadow_recording_20260123_000916_archived_{timestamp}.pkl"

        if src.exists():
            print(f"\n📦 Archiving {src} -> {dst}...")
            shutil.move(str(src), str(dst))
            print("✓ Archived!")
        else:
            print("✗ File not found")

    elif option_id == 3:
        # Reorganize to V1/V2 structure
        print("\n📁 Reorganizing folders...")

        # Create new structure
        Path("data/v1/recordings").mkdir(parents=True, exist_ok=True)
        Path("data/v1/vlm_corrections").mkdir(parents=True, exist_ok=True)
        Path("data/v2/recordings").mkdir(parents=True, exist_ok=True)
        Path("data/v2/vlm_corrections").mkdir(parents=True, exist_ok=True)
        Path("data/yolo/training_data").mkdir(parents=True, exist_ok=True)

        # Move V1 data
        if Path("darkorbit_bot/data/recordings").exists():
            print("  Moving V1 recordings...")
            # Don't move, create symlink or copy
            print("  (Keeping original location, create symlinks manually)")

        # Move V2 data
        if Path("darkorbit_bot/data/recordings_v2").exists():
            print("  Moving V2 recordings...")
            print("  (Keeping original location, create symlinks manually)")

        print("✓ Structure created! (originals left in place)")
        print("\nManually move files if needed:")
        print("  V1: darkorbit_bot/data/recordings -> data/v1/recordings")
        print("  V2: darkorbit_bot/data/recordings_v2 -> data/v2/recordings")

    elif option_id == 4:
        # Clean old checkpoints
        ckpt_dir = Path("darkorbit_bot/data/checkpoints")
        if ckpt_dir.exists():
            print(f"\n🗑️  Deleting {ckpt_dir}...")
            shutil.rmtree(ckpt_dir)
            print("✓ Deleted!")
        else:
            print("✗ Directory not found")

    elif option_id == 5:
        print("\n✓ No changes made")


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("DATA CLEANUP AND ORGANIZATION TOOL")
    print("=" * 80)

    # Step 1: Analyze
    results = analyze_data_folders()

    # Step 2: Identify problems
    problems = identify_problems(results)

    # Step 3: Propose reorganization
    propose_reorganization()

    # Step 4: Cleanup options
    options = cleanup_options()

    # Step 5: Ask user
    print("\n" + "=" * 80)
    choice = input("\nChoose an option (1-5) or 'q' to quit: ").strip()

    if choice.lower() == 'q':
        print("Exiting...")
        return

    try:
        option_id = int(choice)
        if 1 <= option_id <= 5:
            confirm = input(f"\nConfirm option {option_id}? (yes/no): ").strip().lower()
            if confirm == 'yes':
                execute_cleanup(option_id)
            else:
                print("Cancelled")
        else:
            print("Invalid option")
    except ValueError:
        print("Invalid input")


if __name__ == "__main__":
    main()
