import cv2
from PIL import Image
from ultralytics import YOLO
from vit_inferencer import get_emotion

face_model = YOLO('yolov8n-face.pt') #YOLO('yolov8n-face.pt')

# Emotion detection code
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Extract face region
            face_img = frame[y1:y2, x1:x2]
            rgb_frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            
            # Frame is sent to emotion recognition model
            emotion_label, emotion_confidence = get_emotion(pil_frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put emotion label on the bounding box
            label = f"{emotion_label}: {emotion_confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_resized = cv2.resize(frame, (320, 240))
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()