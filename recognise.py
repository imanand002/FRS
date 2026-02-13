# import cv2
# import numpy as np
# import time
# from detector.mediapipe_detector import FaceDetector
# from recognition.facenet import FaceNetRecognizer
# from utils.similarity import cosine_similarity

# # =======================
# # Config
# # =======================
# THRESHOLD = 0.60
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480
# SKIP_FRAMES = 3   # run recognition every 3rd frame

# # =======================
# # Load database
# # =======================
# known_embeddings = np.load("embeddings.npy")
# known_names = np.load("names.npy")

# def normalize(e):
#     return e / (np.linalg.norm(e) + 1e-6)

# known_embeddings = np.array([normalize(e) for e in known_embeddings])

# # =======================
# # Init models
# # =======================
# detector = FaceDetector()
# recognizer = FaceNetRecognizer()

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# # =======================
# # Helpers
# # =======================
# def safe_crop(img, x, y, w, h):
#     padding = 0.3
#     x1 = max(0, int(x - w * padding))
#     y1 = max(0, int(y - h * padding))
#     x2 = min(img.shape[1], int(x + w * (1 + padding)))
#     y2 = min(img.shape[0], int(y + h * (1 + padding)))
#     return img[y1:y2, x1:x2]

# # =======================
# # Main loop
# # =======================
# frame_count = 0
# last_name = "Detecting..."
# last_score = 0.0

# while True:
#     frame_start = time.time()
#     frame_count += 1

#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize for speed
#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#     # Run heavy recognition only every N frames
#     if frame_count % SKIP_FRAMES == 0:
#         result = detector.detect(frame)

#         if result is not None:
#             x, y, w, h, conf = result
#             face = safe_crop(frame, x, y, w, h)

#             if face is not None and face.size > 0:
#                 embedding = normalize(recognizer.get_embedding(face))

#                 # Vectorized cosine similarity
#                 sims = known_embeddings @ embedding
#                 best_idx = int(np.argmax(sims))
#                 best_score = sims[best_idx]
#                 best_match = known_names[best_idx]

#                 if best_score < THRESHOLD:
#                     best_match = "Unknown"

#                 last_name = best_match
#                 last_score = best_score
#                 last_box = (x, y, w, h)

#     # Draw last known result (for smooth display)
#     if 'last_box' in locals():
#         x, y, w, h = last_box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame,
#                     f"{last_name} ({last_score:.2f})",
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0,255,0),
#                     2)

#     # FPS calculation
#     latency = (time.time() - frame_start)
#     fps = 1 / latency if latency > 0 else 0

#     # Print every 10 frames
#     if frame_count % 10 == 0:
#         print(f"Name: {last_name} | Score: {last_score:.2f} | FPS: {fps:.1f}")

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# print("\n[INFO] Exiting...")
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import time
# from deepface import DeepFace
# from detector.mediapipe_detector import FaceDetector
# from recognition.facenet import FaceNetRecognizer
# from utils.similarity import cosine_similarity

# # =======================
# # Config
# # =======================
# THRESHOLD = 0.60
# # Liveness must be True to allow recognition
# LIVENESS_REQUIRED = True 

# # =======================
# # Load models & Data
# # =======================
# detector = FaceDetector()
# recognizer = FaceNetRecognizer()
# known_embeddings = np.load("embeddings.npy")
# known_names = np.load("names.npy")

# def normalize(e):
#     return e / (np.linalg.norm(e) + 1e-6)

# known_embeddings = np.array([normalize(e) for e in known_embeddings])
# cap = cv2.VideoCapture(0)

# print("[INFO] Zero-Tolerance System Active...")

# while True:
#     ret, frame = cap.read()
#     if not ret: break

#     # 1. Detection (MediaPipe is fast)
#     result = detector.detect(frame)

#     if result is not None:
#         x, y, w, h, conf = result
#         face_img = frame[y:y+h, x:x+w]

#         if face_img.size > 0:
#             try:
#                 # 2. ANTI-SPOOFING (The Gatekeeper)
#                 # We analyze the cropped face specifically
#                 objs = DeepFace.extract_faces(
#                     img_path=face_img, 
#                     anti_spoofing=True, 
#                     enforce_detection=False,
#                     detector_backend='skip' # We already detected it
#                 )
                
#                 is_real = objs[0].get("is_real", False)

#                 if is_real:
#                     # 3. RECOGNITION (Only for live humans)
#                     embedding = normalize(recognizer.get_embedding(face_img))
#                     sims = [cosine_similarity(embedding, e) for e in known_embeddings]
#                     best_idx = np.argmax(sims)
                    
#                     if sims[best_idx] >= THRESHOLD:
#                         name = f"{known_names[best_idx]} ({sims[best_idx]:.2f})"
#                         color = (0, 255, 0) # Green
#                     else:
#                         name = "Unknown"
#                         color = (0, 255, 255) # Yellow
#                 else:
#                     name = "SPOOF ATTACK DETECTED"
#                     color = (0, 0, 255) # Red

#             except Exception as e:
#                 name = "Error Analyzing"
#                 color = (255, 255, 255)

#             # UI Overlay
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#             cv2.putText(frame, name, (x, y - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("Secure Face ID", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"): break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import time
from detector.mediapipe_detector import FaceDetector
from recognition.facenet import FaceNetRecognizer

# =======================
# Config
# =======================
THRESHOLD = 0.60
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECOGNITION_INTERVAL = 5   # run embedding every 5 frames

# =======================
# Load database
# =======================
known_embeddings = np.load("embeddings.npy")
known_names = np.load("names.npy")

def normalize(e):
    return e / (np.linalg.norm(e) + 1e-6)

known_embeddings = np.array([normalize(e) for e in known_embeddings])

# =======================
# Init models
# =======================
detector = FaceDetector()
recognizer = FaceNetRecognizer()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

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
frame_count = 0
last_name = "Detecting..."
last_score = 0.0
last_box = None

# FPS smoothing
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_count += 1

    # Always detect face (fast)
    result = detector.detect(frame)

    if result is not None:
        x, y, w, h, conf = result
        last_box = (x, y, w, h)

        # Run heavy recognition only every N frames
        if frame_count % RECOGNITION_INTERVAL == 0:
            face = safe_crop(frame, x, y, w, h)

            if face is not None and face.size > 0:
                embedding = normalize(recognizer.get_embedding(face))
                sims = known_embeddings @ embedding

                best_idx = int(np.argmax(sims))
                best_score = sims[best_idx]
                best_match = known_names[best_idx]

                if best_score < THRESHOLD:
                    best_match = "Unknown"

                last_name = best_match
                last_score = best_score

    # Draw box
    if last_box is not None:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame,
                    f"{last_name} ({last_score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    # Smooth FPS calculation
    current_time = time.time()
    new_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps = (fps * 0.9) + (new_fps * 0.1)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("\n[INFO] Exiting...")
cap.release()
cv2.destroyAllWindows()
