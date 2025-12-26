"""
Import External Dataset for Accident Detection Training

This script imports an external pre-extracted image dataset and prepares it
for training by resizing images and splitting into train/val/test sets.

Source: Kaggle Accident Detection Dataset
Total Images: 13,228 (6,614 accident + 6,614 non-accident)

Usage:
    python import_dataset.py --source "C:\path\to\archive" --output data

Author: Accident Detection Project
Date: December 2025
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import optional dependencies
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Configuration
IMG_SIZE = (224, 224)  # Target size for model input
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
JPEG_QUALITY = 95


def resize_and_copy(src_path: Path, dst_path: Path, resize: bool = True) -> bool:
    """
    Resize an image and save to destination.
    
    Args:
        src_path: Source image path
        dst_path: Destination image path
        resize: Whether to resize the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if resize and HAS_PIL:
            with Image.open(src_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize with high-quality resampling
                img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                img_resized.save(dst_path, 'JPEG', quality=JPEG_QUALITY)
        else:
            # Just copy if no resize needed or PIL not available
            shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def import_dataset(
    source_dir: str,
    output_dir: str,
    resize: bool = True,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    max_workers: int = 4
):
    """
    Import and prepare an external dataset for training.
    
    Args:
        source_dir: Path to source dataset with AccidentData and NonAccidentData folders
        output_dir: Output directory for prepared dataset
        resize: Whether to resize images to 224x224
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data  
        test_ratio: Ratio of test data
        max_workers: Number of parallel workers for processing
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    random.seed(RANDOM_SEED)
    
    print("\n" + "=" * 70)
    print("   DATASET IMPORT TOOL")
    print("   Accident Detection System - Research Paper Dataset Preparation")
    print("=" * 70)
    
    # Detect dataset structure
    accident_dir = None
    non_accident_dir = None
    
    # Check for common naming patterns
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            name_lower = subdir.name.lower()
            if 'accident' in name_lower and 'non' not in name_lower:
                accident_dir = subdir
            elif 'nonaccident' in name_lower or 'non_accident' in name_lower or 'non accident' in name_lower:
                non_accident_dir = subdir
            elif 'normal' in name_lower or 'non' in name_lower:
                non_accident_dir = subdir
    
    if not accident_dir or not non_accident_dir:
        print(f"❌ Could not detect dataset structure in {source_dir}")
        print("   Expected: AccidentData/ and NonAccidentData/ folders")
        return
    
    # Handle nested directories (e.g., AccidentData/AccidentData/)
    if (accident_dir / accident_dir.name).is_dir():
        accident_dir = accident_dir / accident_dir.name
    if (non_accident_dir / non_accident_dir.name).is_dir():
        non_accident_dir = non_accident_dir / non_accident_dir.name
    
    print(f"\n📂 Source Dataset:")
    print(f"   Accident: {accident_dir}")
    print(f"   Non-Accident: {non_accident_dir}")
    
    # Collect all images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    accident_images = [f for f in accident_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in extensions]
    non_accident_images = [f for f in non_accident_dir.iterdir() 
                           if f.is_file() and f.suffix.lower() in extensions]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Accident images: {len(accident_images)}")
    print(f"   Non-Accident images: {len(non_accident_images)}")
    print(f"   Total: {len(accident_images) + len(non_accident_images)}")
    
    # Shuffle for random split
    random.shuffle(accident_images)
    random.shuffle(non_accident_images)
    
    # Calculate splits
    def split_list(images, train_r, val_r, test_r):
        n = len(images)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
    
    accident_splits = split_list(accident_images, train_ratio, val_ratio, test_ratio)
    non_accident_splits = split_list(non_accident_images, train_ratio, val_ratio, test_ratio)
    
    print(f"\n📊 Split Configuration (seed={RANDOM_SEED}):")
    print(f"   Train: {train_ratio:.0%} | Val: {val_ratio:.0%} | Test: {test_ratio:.0%}")
    
    # Create output directories
    class_mapping = {
        'accident': 'Accident',
        'non_accident': 'Non Accident'
    }
    
    for split in ['train', 'val', 'test']:
        for class_name in class_mapping.values():
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process images
    stats = defaultdict(lambda: defaultdict(int))
    
    def process_class(class_key, class_name, splits):
        """Process all images for a class."""
        tasks = []
        for split_name, images in splits.items():
            dst_dir = output_dir / split_name / class_name
            for i, src_path in enumerate(images):
                # Create unique filename
                dst_name = f"{class_key}_{split_name}_{i:05d}.jpg"
                dst_path = dst_dir / dst_name
                tasks.append((src_path, dst_path, split_name))
        return tasks
    
    all_tasks = []
    all_tasks.extend(process_class('accident', 'Accident', accident_splits))
    all_tasks.extend(process_class('nonaccident', 'Non Accident', non_accident_splits))
    
    print(f"\n🔄 Processing {len(all_tasks)} images...")
    if resize:
        print(f"   Resizing to {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
    
    success_count = 0
    error_count = 0
    
    # Progress tracking
    if HAS_TQDM:
        pbar = tqdm(total=len(all_tasks), desc="   Processing", unit="images")
    
    # Process with thread pool for faster I/O
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(resize_and_copy, src, dst, resize): (src, dst, split_name)
            for src, dst, split_name in all_tasks
        }
        
        for future in as_completed(futures):
            src, dst, split_name = futures[future]
            try:
                if future.result():
                    success_count += 1
                    class_name = dst.parent.name
                    stats[split_name][class_name] += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error: {e}")
            
            if HAS_TQDM:
                pbar.update(1)
    
    if HAS_TQDM:
        pbar.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 DATASET IMPORT SUMMARY")
    print("=" * 70)
    
    total_images = 0
    for split in ['train', 'val', 'test']:
        split_total = sum(stats[split].values())
        total_images += split_total
        print(f"\n{split.upper()}: {split_total} images")
        for class_name in sorted(stats[split].keys()):
            print(f"  └─ {class_name}: {stats[split][class_name]}")
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully processed: {success_count} images")
    if error_count > 0:
        print(f"❌ Errors: {error_count} images")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 70)
    
    # Save dataset metadata for reproducibility
    metadata = {
        'source': str(source_dir),
        'total_images': total_images,
        'splits': {
            'train': dict(stats['train']),
            'val': dict(stats['val']),
            'test': dict(stats['test'])
        },
        'config': {
            'random_seed': RANDOM_SEED,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'image_size': IMG_SIZE,
            'resized': resize
        }
    }
    
    metadata_file = output_dir / 'dataset_metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ACCIDENT DETECTION DATASET METADATA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {metadata['source']}\n")
        f.write(f"Total Images: {metadata['total_images']}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}\n\n")
        f.write("SPLIT DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for split in ['train', 'val', 'test']:
            f.write(f"\n{split.upper()}:\n")
            for cls, count in metadata['splits'][split].items():
                f.write(f"  {cls}: {count}\n")
    
    print(f"\n📝 Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Import external dataset for accident detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import with default settings (resize + 70/15/15 split)
  python import_dataset.py --source "C:\\path\\to\\archive" --output data
  
  # Import without resizing
  python import_dataset.py --source "C:\\path\\to\\archive" --output data --no-resize
  
  # Custom split ratios
  python import_dataset.py --source "C:\\path\\to\\archive" --output data --split 80 10 10
        """
    )
    
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source dataset folder")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for prepared dataset")
    parser.add_argument("--no-resize", action="store_true",
                        help="Don't resize images (keep original size)")
    parser.add_argument("--split", type=float, nargs=3, default=[70, 15, 15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Split percentages (default: 70 15 15)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    # Convert percentages to ratios
    ratios = [r / 100 if r > 1 else r for r in args.split]
    
    import_dataset(
        source_dir=args.source,
        output_dir=args.output,
        resize=not args.no_resize,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
