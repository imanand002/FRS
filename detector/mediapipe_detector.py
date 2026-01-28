import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

class FaceDetector:
    def __init__(self, confidence=0.6):
        self.detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence
        )

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        score = detection.score[0]

        h, w, _ = image.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        return (x, y, bw, bh, score)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)

        if result is not None:
            x, y, w, h, conf = result

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw confidence
            cv2.putText(frame, f"{conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
