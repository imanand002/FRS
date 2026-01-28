# import cv2
# import numpy as np
# from detector.mediapipe_detector import FaceDetector
# from recognition.facenet import FaceNetRecognizer
# from utils.similarity import cosine_similarity

# # Load saved embeddings
# known_embeddings = np.load("embeddings.npy")
# known_names = np.load("names.npy")

# detector = FaceDetector()
# recognizer = FaceNetRecognizer()

# THRESHOLD = 0.6

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     result = detector.detect(frame)

#     if result is not None:
#         x, y, w, h, conf = result

#         face = frame[y:y+h, x:x+w]
#         embedding = recognizer.get_embedding(face)

#         best_match = "Unknown"
#         best_score = 0

#         for known_emb, name in zip(known_embeddings, known_names):
#             sim = cosine_similarity(embedding, known_emb)

#             if sim > best_score:
#                 best_score = sim
#                 best_match = name

#         if best_score < THRESHOLD:
#             best_match = "Unknown"

#         # Draw box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

#         # Draw label
#         cv2.putText(frame, f"{best_match} ({best_score:.2f})",
#                     (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0,255,0),
#                     2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from detector.mediapipe_detector import FaceDetector
from recognition.facenet import FaceNetRecognizer
from utils.similarity import cosine_similarity

# Load saved embeddings
known_embeddings = np.load("embeddings.npy")
known_names = np.load("names.npy")

detector = FaceDetector()
recognizer = FaceNetRecognizer()

THRESHOLD = 0.55   # Better for FaceNet than 0.6

cap = cv2.VideoCapture(0)

def safe_crop(img, x, y, w, h):
    padding = 0.3
    x1 = max(0, int(x - w * padding))
    y1 = max(0, int(y - h * padding))
    x2 = min(img.shape[1], int(x + w * (1 + padding)))
    y2 = min(img.shape[0], int(y + h * (1 + padding)))
    return img[y1:y2, x1:x2]

def normalize_embedding(emb):
    return emb / np.linalg.norm(emb)

# Normalize known embeddings once
known_embeddings = np.array([normalize_embedding(e) for e in known_embeddings])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)

    if result is not None:
        x, y, w, h, conf = result

        face = safe_crop(frame, x, y, w, h)

        if face is None or face.size == 0:
            continue

        embedding = recognizer.get_embedding(face)
        embedding = normalize_embedding(embedding)

        best_match = "Unknown"
        best_score = -1

        for known_emb, name in zip(known_embeddings, known_names):
            sim = cosine_similarity(embedding, known_emb)

            if sim > best_score:
                best_score = sim
                best_match = name

        if best_score < THRESHOLD:
            best_match = "Unknown"

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Draw label
        cv2.putText(frame,
                    f"{best_match} ({best_score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
