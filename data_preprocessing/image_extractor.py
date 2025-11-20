#!/usr/bin/env python3
import argparse, os, subprocess, sys, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffprobe_duration(path):

    r = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path)
    ])
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()}")
    return float(r.stdout.strip())

def extract_one(vid_path: Path, root: Path, out_root: Path, fractions, quality=2, epsilon=0.02, overwrite=False):
    rel = vid_path.relative_to(root)
    out_dir = (out_root / rel).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    dur = ffprobe_duration(vid_path)

    if not math.isfinite(dur) or dur <= 0:
        return (vid_path, "skipped (no duration)")

    results = []
    for idx, f in enumerate(fractions, start=1):
        ts = dur * f

        if ts >= dur: ts = max(0.0, dur - epsilon)
        if ts < 0: ts = 0.0
        out_file = out_dir / f"frame_{idx:02d}.jpg"
        if out_file.exists() and not overwrite:
            results.append((idx, "exists"))
            continue
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(vid_path), "-ss", f"{ts:.3f}",
            "-frames:v", "1", "-q:v", str(quality),
            "-y", str(out_file)
        ]
        r = run(cmd)
        if r.returncode != 0:
            results.append((idx, f"error: {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}"))
        else:
            results.append((idx, "ok"))
    return (vid_path, results)

def find_videos(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p

def main():
    ap = argparse.ArgumentParser(description="Extract 5 frames per video using ffmpeg, mirroring folder structure.")
    ap.add_argument("--root", required=True, help="Dataset root (folder containing original_sequences, manipulated_sequences, etc.)")
    ap.add_argument("--out", required=True, help="Output root for extracted frames")
    ap.add_argument("--workers", type=int, default=max(4, os.cpu_count() or 4), help="Parallel workers")
    ap.add_argument("--quality", type=int, default=2, help="JPEG quality 2..31 (lower is better)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 10%, 30%, 50%, 70%, 90%
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]

    vids = list(find_videos(root))
    if not vids:
        print("No videos found under", root)
        sys.exit(1)

    print(f"Found {len(vids)} videos. Extracting frames to {out_root} with {args.workers} workers...")
    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, v, root, out_root, fractions, args.quality, 0.02, args.overwrite) for v in vids]
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                vid_path, results = fut.result()
                if isinstance(results, str):
                    print(f"[{i}/{len(vids)}] {vid_path}: {results}")
                    err += 1
                    continue
                status = ", ".join(f"{idx}:{s}" for idx, s in results)
                if all(s == "ok" or s == "exists" for _, s in results):
                    ok += 1
                else:
                    err += 1
                print(f"[{i}/{len(vids)}] {vid_path} -> {status}")
            except Exception as e:
                print(f"[{i}/{len(vids)}] {e}")
                err += 1
    print(f"Done. Success on {ok}, issues on {err}. Output root: {out_root}")

if __name__ == "__main__":
    main()
