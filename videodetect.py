from ultralytics import YOLO
import cv2

# ── LOAD TRAINED MODEL ─────────────────────────────
model = YOLO(
    r"D:\Github Repositeries\Smart-Suspension\runs\train\suspension_v1\weights\best.pt"
)

# ── VIDEO PATH ─────────────────────────────────────
video_path = r"D:\Github Repositeries\Smart-Suspension\test.mp4"

# ── OPEN VIDEO ─────────────────────────────────────
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error opening video")
    exit()

# ── PROCESS VIDEO FRAME BY FRAME ───────────────────
while True:

    ret, frame = cap.read()

    # End of video
    if not ret:
        break

    # ── RUN YOLO DETECTION ──────────────────────
    results = model.predict(

        source=frame,

        # confidence threshold
        conf=0.05,

        # faster inference
        verbose=False
    )

    # ── DRAW DETECTIONS ─────────────────────────
    annotated_frame = results[0].plot()

    # ── SHOW OUTPUT ─────────────────────────────
    cv2.imshow("Pothole Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── CLEANUP ────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()