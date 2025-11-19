from ultralytics import YOLO
import cv2
from playsound import playsound

model = YOLO("yolo11n.pt")

animal_classes = [
    "cat", "dog", "cow", "horse", "sheep", "elephant",
    "bear", "zebra", "giraffe", "bird"
]

cap = cv2.VideoCapture(0)
animal_found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in animal_classes:
                print(f"Animal Detected: {class_name}")
                animal_found = True

                # Play alert sound (correct usage)
                playsound(r"C:\Users\Naveen karthik\OneDrive\Desktop\animal-detection-software\alert.mp3")

                break

    cv2.imshow("Animal Detector", frame)

    if animal_found:
        print("Stopping loop... Animal detected!")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



