import cv2
import numpy as np
import time

from detector.mediapipe_detector import FaceDetector
from recognition.facenet import FaceNetRecognizer
from utils.similarity import cosine_similarity

# =======================
# Config
# =======================
THRESHOLD = 0.55

# =======================
# Load database
# =======================
known_embeddings = np.load("embeddings.npy")
known_names = np.load("names.npy")

def normalize(e):
    return e / np.linalg.norm(e)

known_embeddings = np.array([normalize(e) for e in known_embeddings])

# =======================
# Init models
# =======================
detector = FaceDetector()
recognizer = FaceNetRecognizer()

cap = cv2.VideoCapture(0)

# =======================
# Helpers
# =======================
def safe_crop(img, x, y, w, h):
    padding = 0.3
    x1 = max(0, int(x - w * padding))
    y1 = max(0, int(y - h * padding))
    x2 = min(img.shape[1], int(x + w * (1 + padding)))
    y2 = min(img.shape[0], int(y + h * (1 + padding)))
    return img[y1:y2, x1:x2]

# =======================
# Main loop
# =======================
while True:
    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)

    if result is not None:
        x, y, w, h, conf = result

        face = safe_crop(frame, x, y, w, h)

        if face is not None and face.size > 0:
            embedding = normalize(recognizer.get_embedding(face))

            sims = [cosine_similarity(embedding, e) for e in known_embeddings]
            best_idx = int(np.argmax(sims))
            best_score = sims[best_idx]
            best_match = known_names[best_idx]

            if best_score < THRESHOLD:
                best_match = "Unknown"

            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame,
                        f"{best_match} ({best_score:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

            # ========== TERMINAL METRICS ==========
            latency = (time.time() - frame_start) * 1000
            fps = 1 / (time.time() - frame_start)

            print(
                f"\rName: {best_match:<10} | "
                f"Similarity: {best_score:.3f} | "
                f"DetectConf: {conf:.2f} | "
                f"Latency: {latency:.1f} ms | "
                f"FPS: {fps:.1f}",
                end=""
            )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("\n[INFO] Exiting...")
cap.release()
cv2.destroyAllWindows()
