import cv2
import torch
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend
from utils.general import scale_boxes
from utils.torch_utils import select_device
# -----------------------------
# CONFIG
# -----------------------------
weights = "yolov9-e-modify-trained.pt"  # your model path
source_video = "bike_video.mp4"  # input video
output_video = "output_bike_tracking.mp4"
img_size = 640
conf_thres = 0.65
iou_thres = 0.45
bike_class_ids = [1]  # adjust according to your dataset class index for 'bike'

device = "" if torch.cuda.is_available() else "cpu"
device = select_device(device)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = DetectMultiBackend(weights, device=device)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, img_size, img_size))

# -----------------------------
# INIT DeepSORT
# -----------------------------
deepsort = DeepSort(max_age=15)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(source_video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_idx = 0
unique_bike_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1]  # BGR -> RGB
    img = torch.from_numpy(img).to(model.device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    # NMS
    from utils.general import non_max_suppression
    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    # -----------------------------
    # PREPARE DeepSORT DETECTIONS
    # -----------------------------
    detections = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in pred:
            if int(cls) in bike_class_ids:
                x1, y1, x2, y2 = [float(x) for x in xyxy]
                w, h = x2 - x1, y2 - y1  # TLWH for DeepSORT
                detections.append(([x1, y1, w, h], float(conf), int(cls)))

    # -----------------------------
    # UPDATE DeepSORT
    # -----------------------------
    tracks = deepsort.update_tracks(detections, frame=frame)

    # -----------------------------
    # DRAW TRACKS
    # -----------------------------
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id
        conf = track.det_conf
        unique_bike_ids.add(track_id)
        label = f"Bike {track_id} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show frame
    cv2.imshow("Bike Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total unique bikes detected: {len(unique_bike_ids)}")
