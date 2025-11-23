import os
import shutil
import argparse
import pandas as pd
import re


def rename_video_folders(base_dir):
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if d == "videos":
                src = os.path.join(root, d)
                dst = os.path.join(root, "images")
                if not os.path.exists(dst):
                    os.rename(src, dst)
                    print(f"[Renamed] {src} → {dst}")
                else:
                    print(f"[Skipped] {dst} already exists")

def get_manipulation_type(filename):
    m = re.search(r"manipulated_sequences_([A-Za-z0-9]+)_", filename)
    if m:
        return m.group(1)
    return "original"

def collect_images_with_labels(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rows = []  # build dataframe rows

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(root, f)

                # Build prefix based on path structure
                parts = src.split(os.sep)
                # expected: base / {original, manipulated} / method / c23 / images / filename
                try:
                    label_folder = parts[1]  # original_sequences / manipulated_sequences
                    method = parts[2]        
                    compression = parts[3]  
                    vid_dir = parts[5]      
                except:
                    label_folder, method, compression, vid_dir = "unknown", "unknown", "unknown", "unknown"

                prefix = f"{label_folder}_{method}_{compression}_{vid_dir}"
                dest_name = f"{prefix}_{f}"
                dest = os.path.join(output_dir, dest_name)
                shutil.copy2(src, dest)

                label = 0 if label_folder == "original_sequences" else 1

                rows.append({
                    "path": dest,
                    "label": label,
                    "method": method,
                })

    df = pd.DataFrame(rows)
    print(f"\n[INFO] Collected {len(df)} images")
    return df

def build_balanced(df, n_real_target=None):
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    if n_real_target is None:
        n_real_target = len(real_df)

    print("Real available:", len(real_df))
    print("Fake available:", len(fake_df))

    # group fakes by manipulation method
    fake_groups = {m: fake_df[fake_df["method"] == m] for m in fake_df["method"].unique()}

    n_methods = len(fake_groups)
    n_per_method = n_real_target // n_methods

    print(f"Sampling {n_per_method} per manipulation method")

    sampled_fakes = []
    for m, g in fake_groups.items():
        sampled = g.sample(n=min(len(g), n_per_method), random_state=42)
        sampled_fakes.append(sampled)

    fake_balanced = pd.concat(sampled_fakes)
    real_balanced = real_df.sample(n=n_real_target, random_state=42)

    df_balanced = pd.concat([real_balanced, fake_balanced]).sample(frac=1, random_state=42)
    print("Final balanced dataset size:", len(df_balanced))
    return df_balanced


def main():
    parser = argparse.ArgumentParser(description="Flatten + Label + Balance FaceForensics++ Dataset")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./all_images")
    parser.add_argument("--balance", action="store_true", help="Build balanced dataset as df_balanced.csv")

    args = parser.parse_args()

    print("\n=== Step 1: Renaming video folders ===")
    rename_video_folders(args.base_dir)

    print("\n=== Step 2: Collecting & labeling images ===")
    df = collect_images_with_labels(args.base_dir, args.output_dir)

    df.to_csv("df_all.csv", index=False)
    print("\nSaved df_all.csv")

    if args.balance:
        print("\n=== Step 3: Building balanced dataset ===")
        df_bal = build_balanced(df)
        df_bal.to_csv("df_balanced.csv", index=False)
        print("Saved df_balanced.csv")


if __name__ == "__main__":
    main()
