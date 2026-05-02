import cv2
import easyocr
from ultralytics import YOLO
import time

# ── Config ──────────────────────────────────────────────
SOURCE = 0            # 0 = webcam | or "path/to/video.mp4"
CONF_THRESHOLD = 0.4  # detection confidence cutoff
# ────────────────────────────────────────────────────────

# Load YOLOv5 model pre-trained on license plates
# This auto-downloads ~14MB on first run
model = YOLO("yolov8n.pt")   # using YOLOv8 (same API, better accuracy)
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Error: Cannot open camera/video.")
    exit()

print("Running... Press Q to quit.")

last_ocr_time = 0
last_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── YOLOv5 / YOLOv8 Detection ──
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]

        # ── OCR (throttled to every 1 second for speed) ──
        now = time.time()
        if now - last_ocr_time > 1.0 and plate_crop.size > 0:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray)
            if ocr_results:
                last_text = max(ocr_results, key=lambda r: r[2])[1].upper()
            last_ocr_time = now

        # ── Draw bounding box ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ── Draw plate text ──
        label = f"{last_text} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()