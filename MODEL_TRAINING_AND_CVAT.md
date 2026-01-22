# Model Training + CVAT Annotation (with Tracking)

This document explains how to build a bicycle detector dataset using **CVAT** (including its **tracking** features), export labels, and train a YOLO model you can use with this repository.

## Goals

- **Annotate bicycles** efficiently in videos using **CVAT interpolation/tracking** tools.
- **Export** annotations to a YOLO-compatible format.
- **Train** a detector and place the resulting weights in `weights/` (e.g. `weights/yolov9e.pt`).

## Prerequisites

- **CVAT** running (self-hosted or cloud).
- A set of **videos** or **images** that match your target camera viewpoint (recommended).
- A clear definition of what counts as a “bike” in your project (bicycle only vs bicycle + rider, etc.).

## Recommended Dataset Design

- **Source diversity**:
  - Day / night
  - Weather variations
  - Different bike types (road, MTB, e-bike, cargo)
  - Occlusions (groups, partial visibility)
- **Split**:
  - Train: ~70–80%
  - Val: ~10–20%
  - Test: ~10% (optional but recommended)
- **Annotation unit**:
  - If you count bikes, label **bicycle bounding boxes** (not riders).
  - Keep boxes tight around the bicycle body and wheels when visible.

## CVAT: Create a Project + Labels

1. Create a **Project** (recommended) so label definitions stay consistent.
2. Add a label:
   - `bicycle` (single class for this repo’s default logic)
3. (Optional) Add extra labels if you plan multi-class training:
   - `person`, `motorbike`, etc.

> Note: This repo’s `bike_counter.py` currently filters detections by a single class id. If you train custom classes, you’ll need to align IDs accordingly.

## CVAT: Create a Task from Video (Best for Tracking)

1. **Create task** → select **Video** upload (or link to cloud storage).
2. Choose an image quality that preserves enough detail for bikes.
3. Set frame step (optional):
   - `1` (best accuracy, most work)
   - `2` or `3` (less work, may miss fast motion)

## CVAT Tracking / Interpolation Workflow (Recommended)

CVAT’s tracking features can dramatically reduce labeling time:

- **Track / Interpolate**:
  - Annotate a bounding box on a key frame.
  - Use CVAT’s **interpolation** across frames (linear) and adjust when it drifts.
- **Keyframes**:
  - Add keyframes whenever the bike changes direction, size, or is partially occluded.
- **Occlusion handling**:
  - If the bike is fully hidden, either stop the track and start a new one later, or mark frames as occluded (depending on your team convention).

### Practical tips for stable tracks

- **Keep IDs consistent** for the same physical bike while visible (helps QA).
- **Avoid “jumping boxes”**:
  - When multiple bikes cross, confirm the track stays on the same bike.
- **Tight boxes**:
  - Avoid oversized boxes that include nearby objects; it reduces detector quality.

## Quality Assurance (QA) Checklist

Before exporting, do a QA pass:

- **No missing bikes** in critical zones (near your counting lines/polygon).
- **No double boxes** on a single bike (unless intentionally multi-class).
- **Consistent label name** (e.g. always `bicycle`, not `bike`, `bicycles`, etc.).
- **Box tightness** consistent across annotators.

## Exporting from CVAT to YOLO Format

CVAT supports exporting in YOLO-like formats, but which one you choose depends on your training code.

### Option A: Export “YOLO” (images + txt labels)

- Export format: **YOLO 1.1** (or similar YOLO bbox export option in your CVAT version).
- You should get:
  - Images
  - A `.txt` label file per image

Each label line typically looks like:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** \([0, 1]\).

### Option B: Export COCO and convert to YOLO

If your CVAT export is COCO JSON, you can convert to YOLO using common scripts (there are many).
This can be more reliable than YOLO export depending on your CVAT version and pipeline.

## Dataset Layout (Recommended)

Create a dataset directory like:

```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

If you exported everything into one folder, you’ll need to split it into train/val and move corresponding label files.

## Training a YOLO Model

This repository contains model definitions under `models/`, but training entrypoints vary by YOLO implementation.
Because different forks/tools have different CLI commands, here is a **general** training checklist that works regardless of which trainer you use:

### 1) Define your classes

- For single-class bike detection:
  - `nc: 1`
  - `names: [bicycle]`

### 2) Create a dataset YAML

Create something like `data/bikes.yaml` (example):

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

nc: 1
names: [bicycle]
```

### 3) Train

Run training using your chosen YOLO trainer (Ultralytics or your YOLOv9 training repo).
Make sure you:

- Use an image size appropriate for your camera (e.g. 640/960/1280)
- Enable augmentation (default usually OK)
- Track metrics on the validation set

### 4) Export weights

When training completes, export or copy the best checkpoint to:

```
weights/yolov9e.pt
```

Then update `bike_counter.py` if your filename differs:

- `MODEL_PATH = "weights/your_file.pt"`

## Aligning Class IDs with `bike_counter.py`

`bike_counter.py` filters detections by `BIKE_CLASS_ID`.

- If you train a **single-class** model where `bicycle` is class 0, set:
  - `BIKE_CLASS_ID = 0`
- If you use COCO-pretrained classes (where `bicycle` is often class 1–3 depending on the model), keep it consistent with your model’s class map.

## Using CVAT Tracking Annotations (If You Need Track IDs)

Training a detector typically **does not require track IDs**—only per-frame bounding boxes.
However, CVAT tracks are useful for:

- Faster labeling via interpolation
- QA (consistent identity over time)
- (Optional) training a dedicated tracker / ReID model later

If your export format includes per-object track IDs, most YOLO detection trainers will simply ignore them.

## Evaluation Tips (Bike Counting Use-Case)

If your goal is accurate counting at lines/polygons (not just mAP), include a targeted validation set:

- Bikes close to the counting lines
- Bikes partially occluded
- Crowded scenes
- Motion blur

Then visually check:

- False positives near the gate/lines (these inflate counts)
- Missed bikes (these reduce counts)

## Optional: Extract Frames from Video (for CVAT or for Training)

If you start from videos, you can extract frames with `ffmpeg`:

```bash
mkdir -p frames
ffmpeg -i input.mp4 -vf fps=15 frames/%06d.jpg
```

Then upload the extracted images to CVAT (image task), or use them directly for training.

## Common Pitfalls

- **Wrong class id** in `bike_counter.py` → no bikes counted.
- **Loose boxes** → detector learns background, increases false positives.
- **Not enough negative examples** (no bikes) → higher false positive rate.
- **Train/val leakage** (same scene in both) → metrics look good but fail in production.

