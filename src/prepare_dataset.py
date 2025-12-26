"""
Dataset Preparation Tool for Accident Detection

This script helps organize and split extracted frames into train/val/test sets.

Usage:
    python prepare_dataset.py --source raw_frames --output data --split 70 15 15
    python prepare_dataset.py --interactive
    
Author: Accident Detection Project
Date: December 2025
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def count_images(directory: Path) -> dict:
    """Count images in each subdirectory."""
    counts = {}
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for subdir in directory.iterdir():
        if subdir.is_dir():
            count = sum(1 for f in subdir.iterdir() 
                       if f.suffix.lower() in extensions)
            counts[subdir.name] = count
    
    return counts


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split images into train/val/test sets.
    
    Args:
        source_dir: Directory with class subdirectories (Accident, Non Accident)
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    random.seed(seed)
    
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"⚠️  Ratios sum to {total:.2f}, normalizing...")
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    print(f"\n📊 Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    
    # Find class directories
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"❌ No class directories found in {source_dir}")
        return
    
    print(f"\n📂 Found {len(class_dirs)} class(es): {[d.name for d in class_dirs]}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_dir in class_dirs:
            (output_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    stats = defaultdict(lambda: defaultdict(int))
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all images
        images = [f for f in class_dir.iterdir() 
                  if f.suffix.lower() in extensions]
        
        if not images:
            print(f"⚠️  No images found in {class_dir}")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split points
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split images
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        print(f"\n📁 {class_name}: {n} images")
        
        # Copy images to split directories
        for split_name, split_images in splits.items():
            dest_dir = output_dir / split_name / class_name
            
            for img_path in split_images:
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                stats[split_name][class_name] += 1
            
            print(f"   {split_name}: {len(split_images)} images")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        total = sum(stats[split].values())
        print(f"\n{split.upper()}: {total} images")
        for class_name, count in stats[split].items():
            print(f"  └─ {class_name}: {count}")
    
    print(f"\n✅ Dataset saved to: {output_dir}")


def merge_existing_data(existing_dir: str, new_data_dir: str, output_dir: str):
    """
    Merge existing dataset with new extracted frames.
    
    Args:
        existing_dir: Path to existing dataset (with train/val/test splits)
        new_data_dir: Path to new data to add
        output_dir: Output directory for merged dataset
    """
    existing_dir = Path(existing_dir)
    new_data_dir = Path(new_data_dir)
    output_dir = Path(output_dir)
    
    print("\n🔀 Merging datasets...")
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # First, copy existing data
    if existing_dir.exists():
        print(f"   Copying existing data from {existing_dir}")
        for split in ['train', 'val', 'test']:
            split_dir = existing_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        dest = output_dir / split / class_dir.name
                        dest.mkdir(parents=True, exist_ok=True)
                        for img in class_dir.iterdir():
                            if img.suffix.lower() in extensions:
                                shutil.copy2(img, dest / img.name)
    
    # Then add new data (split it first)
    print(f"   Adding new data from {new_data_dir}")
    split_dataset(new_data_dir, output_dir)


def show_dataset_stats(data_dir: str):
    """
    Display statistics about the current dataset.
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    print("\n" + "=" * 60)
    print("📊 CURRENT DATASET STATISTICS")
    print("=" * 60)
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            split_total = 0
            print(f"\n{split.upper()}:")
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = sum(1 for f in class_dir.iterdir() 
                               if f.suffix.lower() in extensions)
                    print(f"  └─ {class_dir.name}: {count} images")
                    split_total += count
            print(f"  Total: {split_total} images")
            total_images += split_total
        else:
            print(f"\n{split.upper()}: (not found)")
    
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)


def interactive_mode():
    """Interactive mode for dataset preparation."""
    print("\n" + "=" * 60)
    print("     DATASET PREPARATION TOOL")
    print("     For Accident Detection Training")
    print("=" * 60)
    
    print("\nWhat would you like to do?")
    print("  1. Split raw frames into train/val/test")
    print("  2. View current dataset statistics")
    print("  3. Merge new data with existing dataset")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    script_dir = Path(__file__).parent.parent
    default_data_dir = script_dir / "data"
    
    if choice == "1":
        source = input(f"Enter source directory with class folders\n(e.g., raw_frames/Accident, raw_frames/Non Accident): ").strip().strip('"')
        output = input(f"Enter output directory (default: {default_data_dir}): ").strip().strip('"')
        output = output if output else str(default_data_dir)
        
        train = input("Train ratio (default: 0.70): ").strip()
        val = input("Val ratio (default: 0.15): ").strip()
        test = input("Test ratio (default: 0.15): ").strip()
        
        train = float(train) if train else 0.70
        val = float(val) if val else 0.15
        test = float(test) if test else 0.15
        
        split_dataset(source, output, train, val, test)
        
    elif choice == "2":
        data_dir = input(f"Enter data directory (default: {default_data_dir}): ").strip().strip('"')
        data_dir = data_dir if data_dir else str(default_data_dir)
        show_dataset_stats(data_dir)
        
    elif choice == "3":
        existing = input("Enter existing dataset directory: ").strip().strip('"')
        new_data = input("Enter new data directory: ").strip().strip('"')
        output = input(f"Enter output directory (default: {default_data_dir}): ").strip().strip('"')
        output = output if output else str(default_data_dir)
        merge_existing_data(existing, new_data, output)
    else:
        print("Invalid choice!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and split dataset for accident detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split extracted frames into train/val/test
  python prepare_dataset.py --source raw_frames --output data
  
  # Custom split ratios
  python prepare_dataset.py --source raw_frames --output data --split 80 10 10
  
  # View dataset statistics
  python prepare_dataset.py --stats data
  
  # Interactive mode
  python prepare_dataset.py --interactive
        """
    )
    
    parser.add_argument("--source", type=str, help="Source directory with class subdirectories")
    parser.add_argument("--output", type=str, help="Output directory for split dataset")
    parser.add_argument("--split", type=float, nargs=3, default=[70, 15, 15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Split percentages (default: 70 15 15)")
    parser.add_argument("--stats", type=str, help="Show statistics for dataset directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    if args.stats:
        show_dataset_stats(args.stats)
    elif args.interactive or (args.source is None and args.stats is None):
        interactive_mode()
    elif args.source and args.output:
        # Convert percentages to ratios
        ratios = [r / 100 if r > 1 else r for r in args.split]
        split_dataset(
            args.source, 
            args.output,
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
            seed=args.seed
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
