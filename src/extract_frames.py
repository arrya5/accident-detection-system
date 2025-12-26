"""
Video Frame Extraction Tool for Accident Detection Dataset

This script extracts frames from video files to create training data.
Frames are saved in a format compatible with the training script.

Usage:
    python extract_frames.py --video path/to/video.mp4 --output data/train/Accident --fps 2
    python extract_frames.py --video_dir path/to/videos --output data/train/Accident --fps 2
    
Author: Accident Detection Project
Date: December 2025
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    prefix: str = None,
    resize: tuple = (224, 224),
    skip_blurry: bool = True,
    blur_threshold: float = 100.0
):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 2 fps)
        prefix: Prefix for output filenames (default: video filename)
        resize: Resize frames to this size (width, height) or None to keep original
        skip_blurry: Skip frames that are too blurry
        blur_threshold: Laplacian variance threshold for blur detection
        
    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 0
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"\n📹 Processing: {video_path.name}")
    print(f"   Duration: {duration:.1f}s | FPS: {video_fps:.1f} | Total frames: {total_frames}")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    expected_frames = total_frames // frame_interval
    
    print(f"   Extracting at {fps} fps (every {frame_interval} frames)")
    print(f"   Expected output: ~{expected_frames} frames")
    
    # Set prefix
    if prefix is None:
        prefix = video_path.stem
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    skipped_blur = 0
    
    pbar = tqdm(total=expected_frames, desc="   Extracting", unit="frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only save at specified interval
        if frame_count % frame_interval == 0:
            # Check for blur
            if skip_blurry:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < blur_threshold:
                    skipped_blur += 1
                    frame_count += 1
                    continue
            
            # Resize if specified
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            
            # Save frame
            filename = f"{prefix}_frame_{saved_count:05d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            saved_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"   ✅ Saved: {saved_count} frames")
    if skip_blurry:
        print(f"   ⏭️  Skipped (blurry): {skipped_blur} frames")
    
    return saved_count


def extract_from_directory(
    video_dir: str,
    output_dir: str,
    fps: float = 2.0,
    extensions: tuple = ('.mp4', '.avi', '.mkv', '.mov', '.wmv'),
    **kwargs
):
    """
    Extract frames from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        extensions: Video file extensions to process
        **kwargs: Additional arguments passed to extract_frames_from_video
        
    Returns:
        Total number of frames extracted
    """
    video_dir = Path(video_dir)
    
    # Find all video files
    videos = []
    for ext in extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
        videos.extend(video_dir.glob(f"*{ext.upper()}"))
    
    videos = sorted(set(videos))
    
    if not videos:
        print(f"❌ No video files found in: {video_dir}")
        return 0
    
    print(f"\n🎬 Found {len(videos)} video(s) in {video_dir}")
    print("=" * 60)
    
    total_frames = 0
    for video_path in videos:
        frames = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            **kwargs
        )
        total_frames += frames
    
    print("\n" + "=" * 60)
    print(f"🎉 Total frames extracted: {total_frames}")
    return total_frames


def interactive_mode():
    """
    Interactive mode for easier usage.
    """
    print("\n" + "=" * 60)
    print("     VIDEO FRAME EXTRACTION TOOL")
    print("     For Accident Detection Dataset")
    print("=" * 60)
    
    print("\nChoose extraction mode:")
    print("  1. Extract from a single video")
    print("  2. Extract from all videos in a folder")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        video_path = input("Enter video file path: ").strip().strip('"')
    elif choice == "2":
        video_path = input("Enter video folder path: ").strip().strip('"')
    else:
        print("Invalid choice!")
        return
    
    print("\nChoose output category:")
    print("  1. Accident (for accident videos)")
    print("  2. Non Accident (for normal traffic videos)")
    
    category = input("\nEnter choice (1 or 2): ").strip()
    
    if category == "1":
        category_name = "Accident"
    elif category == "2":
        category_name = "Non Accident"
    else:
        print("Invalid choice!")
        return
    
    print("\nChoose dataset split:")
    print("  1. Train (recommended for most data, ~70%)")
    print("  2. Validation (~15%)")
    print("  3. Test (~15%)")
    
    split = input("\nEnter choice (1, 2, or 3): ").strip()
    
    split_map = {"1": "train", "2": "val", "3": "test"}
    if split not in split_map:
        print("Invalid choice!")
        return
    
    split_name = split_map[split]
    
    # Construct output path
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / split_name / category_name
    
    print(f"\nOutput directory: {output_dir}")
    
    fps = input("Frames per second to extract (default: 2): ").strip()
    fps = float(fps) if fps else 2.0
    
    print("\n" + "-" * 60)
    
    if choice == "1":
        extract_frames_from_video(video_path, output_dir, fps=fps)
    else:
        extract_from_directory(video_path, output_dir, fps=fps)
    
    print("\n✅ Done! You can now run train.py to train with the new data.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for accident detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from a single accident video
  python extract_frames.py --video accident1.mp4 --output data/train/Accident --fps 2
  
  # Extract frames from all videos in a folder
  python extract_frames.py --video_dir ./accident_videos --output data/train/Accident --fps 2
  
  # Extract at higher resolution (no resize)
  python extract_frames.py --video video.mp4 --output data/train/Accident --no-resize
  
  # Interactive mode (guided extraction)
  python extract_frames.py --interactive
        """
    )
    
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Path to directory containing videos")
    parser.add_argument("--output", type=str, required=False, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract (default: 2)")
    parser.add_argument("--no-resize", action="store_true", help="Don't resize frames (keep original size)")
    parser.add_argument("--no-blur-filter", action="store_true", help="Don't skip blurry frames")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Blur detection threshold (default: 100)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or (args.video is None and args.video_dir is None):
        interactive_mode()
        return
    
    # Validate arguments
    if args.video is None and args.video_dir is None:
        parser.error("Either --video or --video_dir is required")
    
    if args.output is None:
        parser.error("--output is required")
    
    resize = None if args.no_resize else (224, 224)
    skip_blurry = not args.no_blur_filter
    
    if args.video:
        extract_frames_from_video(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            resize=resize,
            skip_blurry=skip_blurry,
            blur_threshold=args.blur_threshold
        )
    else:
        extract_from_directory(
            video_dir=args.video_dir,
            output_dir=args.output,
            fps=args.fps,
            resize=resize,
            skip_blurry=skip_blurry,
            blur_threshold=args.blur_threshold
        )


if __name__ == "__main__":
    main()
