# Bike Counter

A computer vision-based bike counting system that uses YOLOv9 object detection and ByteTrack tracking to count bicycles crossing predefined lines and zones in video frames.

## Features

- **Multi-zone counting**: Tracks bikes across three different counting zones:
  - **Line A**: OUT direction only (red line)
  - **Line B**: IN direction only (blue line)
  - **Polygon C**: Gate zone with anti-flicker logic (cyan polygon)
- **Object detection**: Uses YOLOv9-E model for accurate bike detection
- **Multi-object tracking**: ByteTrack algorithm for robust tracking across frames
- **Logical tracking**: Maintains bike identities even when detection temporarily fails
- **Real-time visualization**: Displays detected bikes, tracking IDs, and count statistics
- **Video output**: Optionally saves annotated video with counting results

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for faster inference)
- See `requirements.txt` for full dependency list

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv counterenv
```

### 2. Activate Virtual Environment

```bash
source counterenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

Download the YOLOv9-E model weights and place them in the `weights/` directory:

- **Option 1**: Use the provided `yolov9e.pt` file (if available)
- **Option 2**: Download from [Ultralytics YOLOv9 repository](https://github.com/ultralytics/ultralytics)
- **Option 3**: Download custom trained weights from [fe8kv9 repository](https://github.com/tuansunday05/fe8kv9?tab=readme-ov-file) (yolov9-e-modify-trained.pt)

The model file should be placed at: `weights/yolov9e.pt`

### 5. Prepare Frame Directory

Create a `frames/` directory and add your video frames (as image files) to it. The script processes frames in alphabetical order.

```bash
mkdir frames
# Add your frame images to the frames/ directory
```

## Configuration

Edit the configuration section in `bike_counter.py` to customize the behavior:

### Detection Settings

```python
FRAME_DIR = "frames"              # Directory containing input frames
MODEL_PATH = "weights/yolov9e.pt" # Path to YOLO model weights
BIKE_CLASS_ID = 3                 # COCO class ID for bicycle (3)
CONF_THRES = 0.35                 # Confidence threshold for detections
```

### Tracking Settings

```python
DIST_THRES = 60                   # Maximum distance for track association (pixels)
MAX_MISSING_FRAMES = 15           # Frames before removing lost tracks
MIN_STABLE_FRAMES = 3             # Frames required in polygon C before counting
```

### Video Output Settings

```python
SAVE_VIDEO = True                 # Enable/disable video output
OUTPUT_VIDEO = "bike_counter_result.mp4"  # Output video filename
FPS = 15                          # Output video frame rate
```

### Counting Zones

Adjust the line and polygon coordinates to match your camera view:

```python
# Line A - OUT direction (red)
LINE_A = ((350, 200), (600, 90))

# Line B - IN direction (blue)
LINE_B = ((650, 300), (950, 300))

# Polygon C - Gate zone (cyan)
POLY_C = np.array([
    [450, 600],
    [900, 600],
    [900, 890],
    [450, 890]
], dtype=np.int32)
```

**Note**: Coordinate values are in pixels. Adjust these based on your video resolution and camera angle.

## Usage

### Basic Usage

```bash
python bike_counter.py
```

### During Execution

- The script will display a window showing:
  - Detected bikes with tracking IDs
  - Counting lines and polygon zones
  - Real-time count statistics
- Press **ESC** to stop processing early
- The output video (if enabled) will be saved as `bike_counter_result.mp4`

### Output

The script displays and saves:
- **TOTAL**: Total unique bikes counted (across all zones)
- **Line A (OUT)**: Bikes crossing Line A in OUT direction
- **Line B (IN)**: Bikes crossing Line B in IN direction
- **Gate C**: Bikes entering the polygon gate zone

## How It Works

1. **Frame Processing**: Reads frames sequentially from the `frames/` directory
2. **Object Detection**: YOLOv9 detects bicycles in each frame
3. **Tracking**: ByteTrack maintains track IDs across frames
4. **Logical Tracking**: Custom logic stitches tracks together when ByteTrack IDs change
5. **Line Crossing Detection**: 
   - **Line A/B**: Detects when bike center crosses from one side to the other
   - **Polygon C**: Counts when bike remains inside polygon for `MIN_STABLE_FRAMES` frames
6. **Counting**: Each bike is counted only once per zone to avoid duplicates
7. **Visualization**: Draws detection results, tracking IDs, and statistics on frames

### Counting Logic

- **Line A (OUT)**: Counts when bike moves from positive side to negative side
- **Line B (IN)**: Counts when bike moves from negative side to positive side
- **Polygon C**: Counts when bike is stable inside polygon for minimum frames (anti-flicker)
- **Total Count**: Each bike contributes at most once to the total count

## Troubleshooting

### No detections appearing

- Check that `BIKE_CLASS_ID = 3` matches your model's bicycle class
- Lower `CONF_THRES` if bikes are not being detected
- Verify model weights are correctly loaded

### Incorrect counts

- Adjust `DIST_THRES` if tracks are being lost or incorrectly merged
- Increase `MAX_MISSING_FRAMES` if bikes are disappearing temporarily
- Fine-tune line/polygon coordinates to match your camera view
- Adjust `MIN_STABLE_FRAMES` for polygon C to reduce false positives

### Performance issues

- Use GPU acceleration (CUDA) for faster inference
- Reduce input frame resolution
- Process fewer frames per second

### Video not saving

- Ensure `SAVE_VIDEO = True` in configuration
- Check write permissions for output directory
- Verify video codec support (mp4v)

## File Structure

```
bike_counter/
├── bike_counter.py          # Main script
├── bytetrack.yaml           # ByteTrack tracker configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── weights/
│   └── yolov9e.pt          # YOLO model weights
├── frames/                 # Input frames directory (create this)
└── models/                 # YOLO model architecture definitions
```

## License

This project uses:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking

Please refer to their respective licenses for usage terms.

## Notes

- The script processes frames as image files. To process a video, extract frames first using tools like `ffmpeg`
- Coordinate system: (0,0) is top-left corner, x increases right, y increases down
- Line crossing detection uses signed distance from line to determine direction
- Polygon detection uses OpenCV's point-in-polygon test
