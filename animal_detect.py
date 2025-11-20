from ultralytics import YOLO
import cv2
import pygame
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize pygame for alert sound
pygame.mixer.init()
alert_sound = r"C:\Users\Naveen karthik\OneDrive\Desktop\animal-detection-software\alert.mp3"

# List of animal classes to detect
animal_classes = [
    "cat", "dog", "cow", "horse", "sheep", "elephant",
    "bear", "zebra", "giraffe", "bird"
]

# Load OpenCV human face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

alarm_playing = False
alarm_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # HUMAN FACE DETECTION
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_regions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Human", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        face_regions.append((x, y, x + w, y + h))

    # -------------------------
    # YOLO ANIMAL DETECTION
    # -------------------------
    results = model(frame)

    animal_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Skip if detection overlaps a human face region
            skip = False
            for (fx1, fy1, fx2, fy2) in face_regions:
                if x1 > fx1 and y1 > fy1 and x2 < fx2 and y2 < fy2:
                    skip = True
                    break
            if skip:
                continue

            # If animal detected
            if class_name in animal_classes:
                animal_detected = True

                # Draw red box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{class_name} {conf * 100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                print(f"Animal Detected: {class_name} {conf*100:.1f}%")

    # -------------------------------------------------
    # START ALARM ONLY WHEN ANIMAL IS DETECTED
    # -------------------------------------------------
    if animal_detected and not alarm_playing:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
        alarm_start_time = time.time()
        alarm_playing = True
        print("ALARM STARTED")

    # -------------------------------------------------
    # STOP ALARM AFTER 50 SECONDS
    # -------------------------------------------------
    if alarm_playing:
        elapsed = time.time() - alarm_start_time
        if elapsed >= 10:
            pygame.mixer.music.stop()
            alarm_playing = False
            print("ALARM STOPPED after 50 seconds")

    cv2.imshow("Animal & Human Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
