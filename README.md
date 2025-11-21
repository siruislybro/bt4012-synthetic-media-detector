# BT4012 Fraud Analutics - Synthetic Media Detection

## Introduction
Deepfakes—synthetic media generated using deep learning—pose a growing fraud and security risk. To tackle this, our team proposes a synthetic media detection model with a focus on facial manipulations in image content. 

## Data Preperation & Preprocessing
To address the problem statement and train our detection models effectively, we utilize the FaceForensics++ dataset. 
Source: [FaceForensics++ Github](https://github.com/ondyari/FaceForensics)

The preprocessing workflow consists of three distinct stages:

```mermaid
graph LR
    A[Download Data] -->|faceforensics_download.py| B[Raw Videos]
    B -->|image_extractor.py| C[Extract Frames]
    C -->|flatten_images.py| D[Labeled Flat Dataset]
```

### 1. Dataset Acquisition
We use the FaceForensics++ automated downloader script to acquire the specific subsets of data required (Original sequences, Deepfakes, Face2Face, FaceSwap, etc.).

Source: FaceForensics++ (GitHub/TUM)

Quality: We utilize c23 (Constant Rate Factor) compression to balance quality with realistic web-video artifacts.

**Usage:**
```
Download all datasets with c23 compression
python scripts/faceforensics_download.py ./data/raw_videos \
    -d all \
    -c c23 \
    -t videos \
    --server EU2
```

| Argument | Description                                             |
|----------|---------------------------------------------------------|
| `-d all` | Downloads all subsets (Original, Deepfakes, Face2Face) |
| `-c c23` | Sets compression level (`raw`, `c23`, `c40`)            |
| `-t videos` | Downloads the video files                           |
| `--server EU` | Selects download server (EU/CA)                   |


### 2. Frame Extraction
To enable image-based classification and temporal analysis, we extract discrete frames from the video sequences. Processing whole videos is computationally prohibitive; therefore, we extract a representative distribution of frames.

Sampling Strategy: 5 frames per video are extracted at fixed temporal intervals (10%, 30%, 50%, 70%, and 90% of video duration).

Tooling: Utilizes ffmpeg with multi-threading for high-throughput extraction.

**Usage:**
```
python scripts/image_extractor.py \
    --root ./data/raw_videos \
    --out ./data/extracted_frames \
    --workers 8 \
    --quality 2
```

| Argument   | Description                                                     |
|------------|-----------------------------------------------------------------|
| `--root`   | Directory containing the downloaded FaceForensics++ structure   |
| `--out`    | Destination directory for extracted frames                      |
| `--workers`| Number of parallel threads (defaults to CPU count)              |
| `--quality`| JPEG compression quality for extracted frames (0–3)             |


### 3. Dataset Flattening & Labeling
The raw FaceForensics++ structure is deeply nested. To facilitate efficient data loading (e.g., for PyTorch ImageFolder or custom CSV loaders), we flatten the directory structure and encode metadata directly into the filenames.

The script performs two actions:

Renames recursive videos folders to images.

Aggregates all images into a single directory with a prefix-based naming convention: Label_Method_Compression_VideoID_Frame.jpg.

Usage:
```
python scripts/flatten_images.py \
    --base_dir ./data/extracted_frames \
    --output_dir ./data/image_data_flatten
```

Output Structure: The final directory ./data/image_data_flatten will contain files formatted as:

```
data/image_data_flatten/
    original_sequences_youtube_c23_001_frame_01.jpg
    manipulated_sequences_FaceSwap_c23_001_frame_01.jpg
    manipulated_sequences_Deepfakes_c23_001_frame_03.jpg
```