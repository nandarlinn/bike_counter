import cv2
import os
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
FRAME_DIR = "frames"
MODEL_PATH = "weights/yolov9e.pt"

BIKE_CLASS_ID = 3
CONF_THRES = 0.35

DIST_THRES = 60
MAX_MISSING_FRAMES = 15
MIN_STABLE_FRAMES = 3

SAVE_VIDEO = True
OUTPUT_VIDEO = "bike_counter_result.mp4"
FPS = 15

# =========================
# LINES & POLYGON
# =========================
LINE_A = ((350, 200), (600, 90))     # OUT only
LINE_B = ((650, 300), (950, 300))     # IN only

# Thin polygon gate for Line C
POLY_C = np.array([
    [450, 600],
    [900, 600],
    [900, 890],
    [450, 890]
], dtype=np.int32)

# =========================
# HELPERS
# =========================
def side_of_line(p, line):
    (x1, y1), (x2, y2) = line
    return np.sign((x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1))

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def inside_polygon(p, poly):
    return cv2.pointPolygonTest(poly, p, False) >= 0

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# STATE
# =========================
logical_tracks = {}
next_logical_id = 1
frame_idx = 0

count_A = 0
count_B = 0
count_C = 0
count_TOTAL = 0

video_writer = None

# =========================
# PROCESS FRAMES
# =========================
for fname in sorted(os.listdir(FRAME_DIR)):
    frame_idx += 1
    frame = cv2.imread(os.path.join(FRAME_DIR, fname))
    if frame is None:
        continue

    h, w = frame.shape[:2]

    if SAVE_VIDEO and video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    # =========================
    # TTL CLEANUP
    # =========================
    dead = []
    for lid, t in logical_tracks.items():
        if frame_idx - t["last_seen"] > MAX_MISSING_FRAMES:
            dead.append(lid)
    for lid in dead:
        del logical_tracks[lid]
    
    # =========================
    # YOLO + BYTE
    # =========================
    r = model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

    if r.boxes is None or r.boxes.id is None:
        cv2.imshow("Bike Counter", frame)
        if SAVE_VIDEO:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    clss = r.boxes.cls.cpu().numpy()
    boxes = r.boxes.xyxy.cpu().numpy()

    centers = []
    for cls, box in zip(clss, boxes):
        if int(cls) != BIKE_CLASS_ID:
            continue
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        centers.append((cx, cy))

    # =========================
    # LOGICAL STITCHING
    # =========================
    for c in centers:
        best_id = None
        best_d = DIST_THRES

        for lid, t in logical_tracks.items():
            d = dist(c, t["last_pos"])
            if d < best_d:
                best_d = d
                best_id = lid

        if best_id is None:
            logical_tracks[next_logical_id] = {
                "last_pos": c,
                "last_seen": frame_idx,
                "sideA": side_of_line(c, LINE_A),
                "sideB": side_of_line(c, LINE_B),
                "insideC": inside_polygon(c, POLY_C),
                "stable_C": 1,
                "counted_A": False,
                "counted_B": False,
                "counted_C": False,
                "counted_total": False
            }
            best_id = next_logical_id
            next_logical_id += 1

        t = logical_tracks[best_id]

        prevA, prevB = t["sideA"], t["sideB"]
        curA = side_of_line(c, LINE_A)
        curB = side_of_line(c, LINE_B)

        curInsideC = inside_polygon(c, POLY_C)

        # =========================
        # LINE A → OUT
        # =========================
        if not t["counted_A"] and prevA > 0 and curA < 0:
            count_A += 1
            t["counted_A"] = True
            if not t["counted_total"]:
                count_TOTAL += 1
                t["counted_total"] = True

        # =========================
        # LINE B → IN
        # =========================
        if not t["counted_B"] and prevB < 0 and curB > 0:
            count_B += 1
            t["counted_B"] = True
            if not t["counted_total"]:
                count_TOTAL += 1
                t["counted_total"] = True

        # =========================
        # POLYGON GATE C (ANTI-FLICKER)
        # =========================
        if curInsideC:
            t["stable_C"] += 1
        else:
            t["stable_C"] = 0

        if not t["counted_C"] and t["stable_C"] >= MIN_STABLE_FRAMES:
            count_C += 1
            t["counted_C"] = True
            if not t["counted_total"]:
                count_TOTAL += 1
                t["counted_total"] = True

        # =========================
        # UPDATE
        # =========================
        t["last_pos"] = c
        t["last_seen"] = frame_idx
        t["sideA"], t["sideB"] = curA, curB
        t["insideC"] = curInsideC

        # VISUAL
        color = (0, 255, 0) if t["stable_C"] >= MIN_STABLE_FRAMES else (0, 165, 255)
        cv2.circle(frame, c, 4, color, -1)
        cv2.putText(frame, f"ID {best_id}", (c[0]+5, c[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # =========================
    # DRAW UI
    # =========================
    cv2.line(frame, *LINE_A, (255, 0, 0), 2)
    cv2.line(frame, *LINE_B, (0, 0, 255), 2)
    cv2.polylines(frame, [POLY_C], True, (0, 255, 255), 2)

    cv2.putText(frame, f"TOTAL: {count_TOTAL}", (30, 40), 0, 1.2, (0,255,0), 3)
    cv2.putText(frame, f"Line A (OUT): {count_A}", (30, 80), 0, 1, (255,0,0), 2)
    cv2.putText(frame, f"Line B (IN): {count_B}", (30,120), 0, 1, (0,0,255), 2)
    cv2.putText(frame, f"Gate C: {count_C}", (30,160), 0, 1, (0,255,255), 2)

    cv2.imshow("Bike Counter", frame)
    if SAVE_VIDEO:
        video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
if SAVE_VIDEO and video_writer:
    video_writer.release()

cv2.destroyAllWindows()