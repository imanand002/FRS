# import cv2
# import numpy as np
# import time
# from detector.mediapipe_detector import FaceDetector
# from recognition.facenet import FaceNetRecognizer

# # =======================
# # Config
# # =======================
# THRESHOLD = 0.60
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480
# RECOGNITION_INTERVAL = 5   # run embedding every 5 frames

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
# last_box = None

# # FPS smoothing
# prev_time = time.time()
# fps = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
#     frame_count += 1

#     # Always detect face (fast)
#     result = detector.detect(frame)

#     if result is not None:
#         x, y, w, h, conf = result
#         last_box = (x, y, w, h)

#         # Run heavy recognition only every N frames
#         if frame_count % RECOGNITION_INTERVAL == 0:
#             face = safe_crop(frame, x, y, w, h)

#             if face is not None and face.size > 0:
#                 embedding = normalize(recognizer.get_embedding(face))
#                 sims = known_embeddings @ embedding

#                 best_idx = int(np.argmax(sims))
#                 best_score = sims[best_idx]
#                 best_match = known_names[best_idx]

#                 if best_score < THRESHOLD:
#                     best_match = "Unknown"

#                 last_name = best_match
#                 last_score = best_score

#     # Draw box
#     if last_box is not None:
#         x, y, w, h = last_box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame,
#                     f"{last_name} ({last_score:.2f})",
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0,255,0),
#                     2)

#     # Smooth FPS calculation
#     current_time = time.time()
#     new_fps = 1 / (current_time - prev_time)
#     prev_time = current_time
#     fps = (fps * 0.9) + (new_fps * 0.1)

#     cv2.putText(frame,
#                 f"FPS: {int(fps)}",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0,255,0),
#                 2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# print("\n[INFO] Exiting...")
# cap.release()
# cv2.destroyAllWindows()

#smooth
import cv2
import numpy as np
import time
import threading
from queue import Queue
from detector.mediapipe_detector import FaceDetector
from recognition.facenet import FaceNetRecognizer

# =======================
# Config
# =======================
THRESHOLD = 0.60
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Recognition happens in the background; this controls how often we 
# send a new face to the background thread.
RECOGNITION_DELAY = 0.2  # Seconds

# =======================
# Load database
# =======================
known_embeddings = np.load("embeddings.npy").astype('float32')
known_names = np.load("names.npy")

def normalize(e):
    return e / (np.linalg.norm(e) + 1e-6)

known_embeddings = np.array([normalize(e) for e in known_embeddings])

# =======================
# Init models & Shared State
# =======================
detector = FaceDetector()
recognizer = FaceNetRecognizer()

# Shared variables between threads
latest_result = {"name": "Detecting...", "score": 0.0}
recognition_queue = Queue(maxsize=1) 

def recognition_worker():
    """Background thread for heavy inference."""
    global latest_result
    while True:
        face_crop = recognition_queue.get()
        if face_crop is None: break
        
        # Inference
        embedding = normalize(recognizer.get_embedding(face_crop))
        sims = known_embeddings @ embedding
        
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        
        if best_score >= THRESHOLD:
            name = known_names[best_idx]
        else:
            name = "Unknown"
            
        latest_result = {"name": name, "score": best_score}
        recognition_queue.task_done()

# Start the worker thread
worker = threading.Thread(target=recognition_worker, daemon=True)
worker.start()

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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

last_recog_time = 0
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Fast Detection (Main Thread)
    result = detector.detect(frame)
    
    if result is not None:
        x, y, w, h, conf = result
        
        # 2. Trigger Recognition (Async)
        current_time = time.time()
        if (current_time - last_recog_time) > RECOGNITION_DELAY:
            if recognition_queue.empty(): # Only send if worker is free
                face = safe_crop(frame, x, y, w, h)
                if face.size > 0:
                    recognition_queue.put(face)
                    last_recog_time = current_time

        # 3. Draw Results (Using latest data from worker)
        display_name = latest_result["name"]
        display_score = latest_result["score"]
        
        color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{display_name} ({display_score:.2f})", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS Calc
    curr_time = time.time()
    fps = (fps * 0.9) + ((1 / (curr_time - prev_time)) * 0.1)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Fast Face Rec", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()