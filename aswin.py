import cv2
from ultralytics import YOLO
import math 

# Load YOLOv8 model
model = YOLO('best.pt')  # Ganti dengan jalur yang sesuai ke model YOLOv8 Anda

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# object classes
classNames = ["Boredom",
              "Confusion",
              "Engaged",
              "Frustration",
              "Sleepy",
              "Yawning"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            if confidence >= 0.7:  # Treshold 0.7
                # class name
                cls = int(box.cls[0])

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
