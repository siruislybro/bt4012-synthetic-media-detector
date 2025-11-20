import os
import shutil
import argparse

def rename_video_folders(base_dir):
    """Rename all 'videos' folders to 'images' recursively."""
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if d == "videos":
                video_folder = os.path.join(root, d)
                image_folder = os.path.join(root, "images")

                if not os.path.exists(image_folder):
                    os.rename(video_folder, image_folder)
                    print(f"[Renamed] {video_folder} → {image_folder}")
                else:
                    print(f"[Skipped] {image_folder} already exists")


def collect_images(base_dir, output_dir):
    """Copy and rename all images from 'images' folders into a single directory."""
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(root, f)

                # Example path: ./manipulated_sequences/FaceSwap/c23/images/00001.jpg
                parts = src.split(os.sep)
                if len(parts) >= 5:
                    label = parts[1]  # manipulated_sequences / original_sequences
                    sub1 = parts[2]   # e.g. FaceSwap / youtube
                    sub2 = parts[3]   # e.g. c23
                    sub3 = parts[5]
                    prefix = f"{label}_{sub1}_{sub2}_{sub3}"
                else:
                    prefix = "unknown"

                dest_name = f"{prefix}_{f}"
                dest = os.path.join(output_dir, dest_name)

                shutil.copy2(src, dest)
                print(f"[Copied] {src} → {dest}")

    print(f"\n✅ All images collected into: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten all DeepFake/Original images into a single labeled directory."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing manipulated_sequences and original_sequences folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./all_images",
        help="Output directory for collected images (default: ./all_images)",
    )

    args = parser.parse_args()

    print("\n=== Step 1: Renaming 'videos' to 'images' ===")
    rename_video_folders(args.base_dir)

    print("\n=== Step 2: Collecting and labeling images ===")
    collect_images(args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()

