import cv2
import torch
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import colors

# -----------------------------
# CONFIG
# -----------------------------
YOLO_WEIGHTS = "yolov9-e-modify-trained.pt"
SOURCE_FOLDER = "frames/"          # folder with video frames as images
OUTPUT_VIDEO = "bike_tracking_output.mp4"
BIKE_CLASS_NAME = "bike"
IMG_SIZE = 640
CONF_THRESH = 0.65
IOU_THRESH = 0.45
MIN_CONF_TO_COUNT = 0.5             # minimum confidence to count the bike
FPS = 20
# -----------------------------

def main():
    # Select device
    device = select_device("" if torch.cuda.is_available() else "cpu")

    # Load YOLOv9e model
    model = DetectMultiBackend(YOLO_WEIGHTS, device=device)
    stride, names = model.stride, model.names

    # Identify bike class IDs
    bike_class_ids = [i for i, name in names.items() if BIKE_CLASS_NAME in name.lower()]
    if not bike_class_ids:
        raise ValueError(f"No class found for '{BIKE_CLASS_NAME}' in model names: {names}")

    # Initialize DeepSORT tracker
    deepsort = DeepSort(max_age=15)

    # Collect frame paths
    frame_paths = sorted(Path(SOURCE_FOLDER).glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {SOURCE_FOLDER}")

    # Prepare output video
    first_frame = cv2.imread(str(frame_paths[0]))
    H, W = first_frame.shape[:2]
    out_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # Track unique bikes
    unique_bike_ids = set()

    for fpath in frame_paths:
        img0 = cv2.imread(str(fpath))
        if img0 is None:
            continue

        # -----------------------------
        # YOLO PREPROCESS & INFERENCE
        # -----------------------------
        img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1].copy()  # BGR -> RGB
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]

        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)[0]

        # -----------------------------
        # PREPARE DeepSORT DETECTIONS
        # -----------------------------
        deepsort_inputs = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred:
                if int(cls) in bike_class_ids and float(conf) >= MIN_CONF_TO_COUNT:
                    x1, y1, x2, y2 = map(float, xyxy)
                    w, h = x2 - x1, y2 - y1
                    deepsort_inputs.append(([x1, y1, w, h], float(conf), int(cls)))

        # -----------------------------
        # UPDATE DeepSORT
        # -----------------------------
        tracks = deepsort.update_tracks(deepsort_inputs, frame=img0)

        # -----------------------------
        # DRAW TRACKS & COUNT
        # -----------------------------
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # Safe confidence
            conf = track.det_conf if track.det_conf is not None else 0.0

            # Only count if above min confidence
            if conf >= MIN_CONF_TO_COUNT:
                unique_bike_ids.add(track_id)

            label = f"Bike {track_id} ({conf:.2f})"
            color = colors(track_id, True)
            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img0, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlay total bike count
        cv2.putText(img0, f"Total Bikes: {len(unique_bike_ids)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Write frame to video
        out_video.write(img0)

        # Optional display
        cv2.imshow("Bike Tracker", img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------
    # CLEANUP
    # -----------------------------
    out_video.release()
    cv2.destroyAllWindows()
    print(f"\nFINAL UNIQUE BIKE COUNT: {len(unique_bike_ids)}")
    print(f"Video saved as: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()

# import cv2
# import torch
# from pathlib import Path
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from utils.torch_utils import select_device
# from models.common import DetectMultiBackend
# from utils.general import non_max_suppression, scale_boxes
# from utils.plots import colors
# import math
# import numpy as np

# # -----------------------------
# # CONFIG
# # -----------------------------
# YOLO_WEIGHTS = "yolov9-e-modify-trained.pt"
# SOURCE_FOLDER = "frames/"          # folder with video frames as images
# OUTPUT_VIDEO = "bike_tracking_output.mp4"
# BIKE_CLASS_NAME = "bike"
# IMG_SIZE = 640
# CONF_THRESH = 0.65
# IOU_THRESH = 0.45
# MIN_CONF_TO_COUNT = 0.5
# FPS = 20

# # Fisheye correction settings (optional)
# APPLY_FISHEYE_CORRECTION = False
# K = np.array([[300, 0, IMG_SIZE//2], [0, 300, IMG_SIZE//2], [0,0,1]])  # example intrinsic
# D = np.array([-0.2, 0.1, 0, 0])  # example distortion coefficients
# # -----------------------------

# def undistort_frame(frame):
#     h, w = frame.shape[:2]
#     new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1)
#     return cv2.undistort(frame, K, D, None, new_K)

# def is_new_bike(center, existing_centers, min_dist=50):
#     """Return True if the center is far from existing bike centers."""
#     for c in existing_centers:
#         if math.hypot(center[0]-c[0], center[1]-c[1]) < min_dist:
#             return False
#     return True

# def main():
#     device = select_device("" if torch.cuda.is_available() else "cpu")
#     model = DetectMultiBackend(YOLO_WEIGHTS, device=device)
#     stride, names = model.stride, model.names

#     # Identify bike class IDs
#     bike_class_ids = [i for i, name in names.items() if BIKE_CLASS_NAME in name.lower()]
#     if not bike_class_ids:
#         raise ValueError(f"No class found for '{BIKE_CLASS_NAME}' in model names: {names}")

#     # Initialize DeepSORT with frame embeddings
#     deepsort = DeepSort(max_age=30, embedder="mobilenet")  # longer max_age improves tracking

#     # Collect frame paths
#     frame_paths = sorted(Path(SOURCE_FOLDER).glob("*.jpg"))
#     if not frame_paths:
#         raise FileNotFoundError(f"No frames found in {SOURCE_FOLDER}")

#     first_frame = cv2.imread(str(frame_paths[0]))
#     H, W = first_frame.shape[:2]
#     out_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

#     unique_bike_ids = set()
#     bike_centers = []

#     for fpath in frame_paths:
#         img0 = cv2.imread(str(fpath))
#         if img0 is None:
#             continue

#         # Optional fisheye correction
#         if APPLY_FISHEYE_CORRECTION:
#             img0 = undistort_frame(img0)

#         # YOLO preprocessing
#         img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE))
#         img = img[:, :, ::-1].copy()  # BGR → RGB
#         img_tensor = torch.from_numpy(img).to(device).float() / 255.0
#         img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

#         with torch.no_grad():
#             pred = model(img_tensor)[0]

#         pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)[0]

#         deepsort_inputs = []
#         if pred is not None and len(pred):
#             pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
#             for *xyxy, conf, cls in pred:
#                 if int(cls) in bike_class_ids and float(conf) >= MIN_CONF_TO_COUNT:
#                     x1, y1, x2, y2 = map(float, xyxy)
#                     w, h = x2 - x1, y2 - y1
#                     deepsort_inputs.append(([x1, y1, w, h], float(conf), int(cls)))

#         # Update tracker with frame embeddings
#         tracks = deepsort.update_tracks(deepsort_inputs, frame=img0)

#         bike_centers_this_frame = []

#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             track_id = track.track_id
#             x1, y1, x2, y2 = map(int, track.to_ltrb())
#             conf = track.det_conf if track.det_conf is not None else 0.0

#             center = ((x1+x2)//2, (y1+y2)//2)

#             # Count bike only if far from existing centers (reduces duplicates)
#             if conf >= MIN_CONF_TO_COUNT and is_new_bike(center, bike_centers, min_dist=50):
#                 unique_bike_ids.add(track_id)
#                 bike_centers.append(center)

#             label = f"Bike {track_id} ({conf:.2f})"
#             color = colors(track_id, True)
#             cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img0, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Overlay total bike count
#         cv2.putText(img0, f"Total Bikes: {len(unique_bike_ids)}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

#         out_video.write(img0)
#         cv2.imshow("Bike Tracker", img0)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     out_video.release()
#     cv2.destroyAllWindows()
#     print(f"\nFINAL UNIQUE BIKE COUNT: {len(unique_bike_ids)}")
#     print(f"Video saved as: {OUTPUT_VIDEO}")


# if __name__ == "__main__":
#     main()
