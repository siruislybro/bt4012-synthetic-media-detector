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


### 3. Dataset Flattening , Labeling & Rebalancing
The raw FaceForensics++ structure is deeply nested. To facilitate efficient data loading (e.g., for PyTorch ImageFolder or custom CSV loaders), we flatten the directory structure and encode metadata directly into the filenames.

The script performs three actions:

Renames recursive videos folders to images.

Aggregates all images into a single directory with a prefix-based naming convention: Label_Method_Compression_VideoID_Frame.jpg. Also samples to create balanced real v.s. synthetic distribution

Usage:
```
python scripts/flatten_images.py \
    --base_dir ./data/extracted_frames \
    --output_dir ./data/image_data_flatten \
    --balance
```

Output Structure: The final directory ./data/image_data_flatten will contain files formatted as:

```
data/image_data_flatten/
    original_sequences_youtube_c23_001_frame_01.jpg
    manipulated_sequences_FaceSwap_c23_001_frame_01.jpg
    manipulated_sequences_Deepfakes_c23_001_frame_03.jpg
```

A sample of this dataset can be found in `/data/image_sample_flatten`. The full dataset of extracted frames can be found in [Google Drive](https://drive.google.com/drive/folders/1tzRkgLZoTpRiwtFWVlLgIIf_CgKUYLmU?usp=sharing)

## Modelling & Experiments 

The main notebook ties the whole pipeline together and runs both classical and deep learning experiments on the flattened image dataset.


### 1. Data loading and splits
The notebook reads a CSV of the flattened frames (path, label, method), builds a method balanced training set (real vs fake and across manipulation types), and creates stratified train, validation, and test splits.


### 2. Handcrafted forensic features 
For each image, the notebook loads the face centred crop and computes a compact set of forensic features (high pass filter statistics, edge density, gradient energy, LBP uniformity, DCT band ratios, JPEG blockiness, ring vs centre sharpness/brightness/chroma).


### 3. Classical Baselines
The feature vectors are used to train classical models such as logistic regression and tree based ensembles. We report baseline metrics (accuracy, precision, recall, F1, ROC AUC) to understand how far we can go with purely handcrafted features.

### 4. Fusion CNN model in PyTorch
A ResNet50 backbone (ImageNet pretrained) is used to extract deep image features, which are concatenated with the handcrafted feature vector and passed through a small multilayer perceptron head. The model is trained with class weighted binary cross entropy and evaluated using confusion matrix, accuracy, precision, recall, F1, and ROC AUC.


### 5. Xception baseline in Keras
As a comparison, the notebook also trains a pure Xception based image classifier using Keras generators on the same splits, with class weights, early stopping, and model checkpointing on validation AUC.


### 6. Error analysis by manipulation method
Finally, predictions are merged back with the test metadata to compute per method performance (Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter, youtube), highlighting which manipulation types are easier or harder to detect and whether the models generalise beyond the training distribution.

