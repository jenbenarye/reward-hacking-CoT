#!/usr/bin/env python3
"""
Helper script to check the number of samples in each dataset split.
"""

from impossiblebench import impossible_swebench

def main():
    splits = ['original', 'oneoff', 'conflicting']
    
    print("Checking dataset sizes for each split...")
    print("=" * 60)
    
    for split in splits:
        try:
            # Create task without limit to get full dataset size
            task = impossible_swebench(split=split, limit=None, shuffle=False)
            size = len(task.dataset)
            print(f"{split:12s}: {size:6d} samples")
        except Exception as e:
            print(f"{split:12s}: Error - {e}")
    
    print("=" * 60)
    print("\nNote: These are the total number of samples in each split.")
    print("Use --limit to restrict the number of samples when running evaluations.")

if __name__ == "__main__":
    main()

