import cv2
from detector.mediapipe_detector import FaceDetector
from recognition.facenet_model import FaceNetRecognizer
from utils.similarity import cosine_similarity

detector = FaceDetector()
recognizer = FaceNetRecognizer()

# Example: store one known face
known_embedding = None

cap = cv2.VideoCapture(0)

print("Press 's' to save face as known")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detector.detect_and_crop(frame)

    if face is not None:
        emb = recognizer.get_embedding(face)

        if known_embedding is not None:
            sim = cosine_similarity(known_embedding, emb)
            cv2.putText(frame, f"Similarity: {sim:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FaceNet Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and face is not None:
        known_embedding = recognizer.get_embedding(face)
        print("Known face saved")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
