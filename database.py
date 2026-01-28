import os
import cv2
import numpy as np
from detector.mediapipe_detector import FaceDetector
from recognition.facenet import FaceNetRecognizer
from utils.similarity import cosine_similarity

DATASET_PATH = "/Users/mayankanand/Desktop/Anand_coding/Vishveya/FRS/dataset"

detector = FaceDetector()
recognizer = FaceNetRecognizer()

known_embeddings = []
known_names = []

def safe_crop(img, x, y, w, h):
    padding = 0.35
    x1 = max(0, int(x - w * padding))
    y1 = max(0, int(y - h * padding))
    x2 = min(img.shape[1], int(x + w * (1 + padding)))
    y2 = min(img.shape[0], int(y + h * (1 + padding)))
    return img[y1:y2, x1:x2]

print("[INFO] Building face database...")

for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        result = detector.detect(img)
        if result is None:
            continue

        x, y, w, h, conf = result
        face = safe_crop(img, x, y, w, h)

        if face.size == 0:
            continue

        # Show cropped face (debug)
        cv2.imshow("Cropped face", face)
        cv2.waitKey(100)

        embedding = recognizer.get_embedding(face)

        known_embeddings.append(embedding)
        known_names.append(person_name)

cv2.destroyAllWindows()

print(f"[INFO] Loaded {len(known_embeddings)} face samples")

# Save database
np.save("embeddings.npy", known_embeddings)
np.save("names.npy", known_names)

print("[INFO] Database saved (embeddings.npy, names.npy)")

# Debug similarity test
print("\n[DEBUG] Testing embedding consistency...")

if len(known_embeddings) >= 2:
    print("Same image vs same image:",
          cosine_similarity(known_embeddings[0], known_embeddings[0]))

    print("Image 0 vs image 1:",
          cosine_similarity(known_embeddings[0], known_embeddings[1]))
