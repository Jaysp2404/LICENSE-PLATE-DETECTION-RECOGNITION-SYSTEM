import cv2
import easyocr
from ultralytics import YOLO
import time
import numpy as np
from collections import Counter

# ── Config ──────────────────────────────────────────────
SOURCE = 0              # 0 = webcam | "path/to/video.mp4"
CONF_THRESHOLD = 0.35   # lower = detect more, higher = stricter
SAVE_OUTPUT = True      # saves output.mp4
SHOW_WINDOW = True      # set False if cv2.imshow fails
# ────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Error: Cannot open camera/video.")
    exit()

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 20

out = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, FPS, (W, H))
    print("Recording to output.mp4")

print("Running... Press Q to quit.")

# ── OCR state ──
last_ocr_time = 0
plate_history = []       # stores last N readings for voting
HISTORY_LEN = 7          # vote over last 7 readings
OCR_INTERVAL = 0.6       # run OCR every 0.6s

def preprocess_plate(crop):
    """Multi-step image enhancement for better OCR accuracy."""
    # 1. Upscale small plates (OCR works better on larger images)
    h, w = crop.shape[:2]
    scale = max(1, 200 // max(h, 1))
    crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE — adaptive contrast enhancement (handles shadows/glare)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Bilateral filter — reduces noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # 5. Otsu thresholding — auto binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def clean_plate_text(text):
    """Remove junk characters commonly misread by OCR."""
    import re
    text = text.upper().strip()
    # Keep only alphanumeric + spaces + hyphens
    text = re.sub(r'[^A-Z0-9\s\-]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def vote_best_plate(history):
    """Return the most commonly seen plate text (majority vote)."""
    if not history:
        return ""
    counts = Counter(history)
    return counts.most_common(1)[0][0]

def run_ocr(plate_crop):
    """Run EasyOCR with preprocessing and return cleaned text."""
    if plate_crop is None or plate_crop.size == 0:
        return ""
    try:
        processed = preprocess_plate(plate_crop)
        # allowlist = only chars that appear on license plates
        results = reader.readtext(
            processed,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
            detail=1,
            paragraph=False
        )
        if not results:
            return ""
        # Pick highest confidence result
        best = max(results, key=lambda r: r[2])
        conf = best[2]
        text = best[1]
        if conf < 0.3:   # ignore very low confidence OCR
            return ""
        return clean_plate_text(text)
    except Exception as e:
        return ""

def expand_box(x1, y1, x2, y2, W, H, pad=6):
    """Slightly expand bounding box to capture full plate."""
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(W, x2 + pad),
        min(H, y2 + pad)
    )

frame_count = 0
fps_time = time.time()
display_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── FPS counter ──
    if frame_count % 15 == 0:
        display_fps = 15 / (time.time() - fps_time)
        fps_time = time.time()

    # ── YOLOv8 Detection (higher resolution input = better small plate detection) ──
    results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    detected_any = False

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H)

        plate_crop = frame[y1:y2, x1:x2]
        detected_any = True

        # ── OCR throttled by time interval ──
        now = time.time()
        if now - last_ocr_time > OCR_INTERVAL:
            text = run_ocr(plate_crop)
            if text:
                plate_history.append(text)
                if len(plate_history) > HISTORY_LEN:
                    plate_history.pop(0)
            last_ocr_time = now

        # ── Get voted best plate ──
        best_plate = vote_best_plate(plate_history)

        # ── Draw green bounding box ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Draw label background + text ──
        label = f"{best_plate}  [{conf:.2f}]" if best_plate else f"Plate [{conf:.2f}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # ── Print to console ──
        if best_plate:
            print(f"[Frame {frame_count}] Plate: {best_plate}  Confidence: {conf:.2f}")

    # ── HUD overlay ──
    cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Detections: {len(results.boxes)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    status = "DETECTING" if detected_any else "Scanning..."
    color = (0, 255, 0) if detected_any else (0, 165, 255)
    cv2.putText(frame, status, (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if out:
        out.write(frame)

    if SHOW_WINDOW:
        try:
            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            print("Window display failed — output being saved to output.mp4")
            SHOW_WINDOW = False

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print(f"\nDone! Processed {frame_count} frames.")
if SAVE_OUTPUT:
    print("Saved: output.mp4")