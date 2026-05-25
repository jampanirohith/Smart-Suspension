from ultralytics import YOLO
import cv2

# ── LOAD TRAINED MODEL ─────────────────────────────────
model = YOLO(
    r"D:\Github Repositeries\Smart-Suspension\runs\train\suspension_v1\weights\best.pt"
)

# ── IMAGE PATH ─────────────────────────────────────────
image_path = (
    r"D:\Github Repositeries\Smart-Suspension\pothole-small-1\train\images"
    r"\1_jpg.rf.64cd4a8418715dae0c54f9b1c26c4dd5.jpg"
)

# ── RUN PREDICTION ─────────────────────────────────────
results = model.predict(

    source=image_path,

    # Lower confidence for weakly trained model
    conf=0.05,

    # Show image window
    show=True,

    # Save output image
    save=True,

    # Better console output
    verbose=True
)

# ── PRINT DETECTIONS ───────────────────────────────────
for result in results:

    boxes = result.boxes

    print(f"\nDetected Objects: {len(boxes)}")

    for box in boxes:

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        print(f"Class ID: {class_id}")
        print(f"Confidence: {confidence:.2f}")

# ── KEEP WINDOW OPEN ───────────────────────────────────
cv2.waitKey(0)
cv2.destroyAllWindows()