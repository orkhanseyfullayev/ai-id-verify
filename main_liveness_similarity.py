import cv2
import os
import time
import torch
from numpy import dot
from numpy.linalg import norm
from sixdrepnet import regressor
from insightface.app import FaceAnalysis
import json

# === Helper Function: Capture Photo ===
def capture_image(filename, message):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera could not be opened. Try using index 1 or 2 if another application is using it.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.rectangle(frame, (25, 10), (550, 40), (200, 200, 200), -1)    
        cv2.putText(frame, message, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            cv2.imwrite(filename, frame)
            print(f"{filename} saved.")
            break
        elif key & 0xFF == 27:
            break
        elif key & 0xFF == ord('q'):
            print("Operation cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

# === Step 1: Capture Photos ===
os.makedirs("images_similarity", exist_ok=True)
capture_image("images_similarity/face.png", "Press 's' to capture your face.")
capture_image("images_similarity/id.png", "Press 's' to capture your ID photo.")

# === Step 2: Liveness Detection ===
print("\nStarting liveness verification...")
detector = regressor.SixDRepNet_Detector(
    gpu_id=0,
    dict_path="6DRepNet_300W_LP_AFLW2000.pth"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

stage = "right"
right_done = False
left_done = False
start_time = time.time()
TOTAL_TIME = 10
liveness_passed = False
liveness_timeout = False

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    elapsed = int(time.time() - start_time)
    remaining = max(TOTAL_TIME - elapsed, 0)

    try:
        pitch, yaw, roll = detector.predict(frame)
        frame = detector.draw_axis(frame, yaw, pitch, roll)

        cv2.putText(frame, f"Yaw:{int(yaw)} Pitch:{int(pitch)} Roll:{int(roll)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if not liveness_passed:
            cv2.rectangle(frame, (470, 10), (590, 40), (200, 200, 200), -1)
            cv2.putText(frame, f"Time: {remaining}s", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if remaining == 0 and not liveness_passed:
            liveness_timeout = True
            cv2.putText(frame, "Time expired", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            if stage == "right":
                cv2.putText(frame, "Please turn your head RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if yaw > 20:
                    right_done = True
                    stage = "left"
            elif stage == "left":
                cv2.putText(frame, "Please turn your head LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if yaw < -20:
                    left_done = True
                    stage = "done"
            elif stage == "done":
                if not liveness_passed:
                    liveness_passed = True
                cv2.putText(frame, "Liveness confirmed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to exit", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    except Exception as e:
        cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Liveness Test", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        if liveness_passed or liveness_timeout:
            break

cap.release()
cv2.destroyAllWindows()

# === Step 3: Collect and Save Results ===
similarity = -1
result = "not_checked"

if liveness_passed:
    print("\nLiveness Result: Successful")
    print("\nStarting Similarity Analysis...")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    img1 = cv2.imread("images_similarity/face.png")
    img2 = cv2.imread("images_similarity/id.png")

    faces1 = app.get(img1)
    faces2 = app.get(img2)

    if not faces1 or not faces2:
        result = "not_detected"
    else:
        emb1 = faces1[0].normed_embedding
        emb2 = faces2[0].normed_embedding

        similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        score = round(similarity * 100)

        if similarity >= 0.50:
            result = "matched"
        elif 0.40 <= similarity < 0.50:
            result = "suspicious"
        else:
            result = "not_matched"

# Save results as JSON
os.makedirs("result", exist_ok=True)
with open("liveness_result.json", "w", encoding="utf-8") as f:
    json.dump({
        "liveness_result": "passed" if liveness_passed else "failed",
        "similarity_result": result,
        "score": score if liveness_passed and 'score' in locals() else None
    }, f, indent=2, ensure_ascii=False)
