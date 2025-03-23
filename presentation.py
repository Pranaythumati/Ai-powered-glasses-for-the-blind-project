#code for selected object
import cv2
import torch
import pyttsx3
import numpy as np
import threading
import speech_recognition as sr

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open webcam
cap = cv2.VideoCapture(1)

# Set a threshold to filter weak detections
CONFIDENCE_THRESHOLD = 0.5

# Store previously detected objects to prevent redundant announcements
previous_detections = set()

# Variable to store the user-specified object
target_object = None

def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    """Continuously listens for the user's voice command to specify an object."""
    global target_object
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for object name...")
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                print("Say the name of the object you want to detect:")
                audio = recognizer.listen(source)
            detected_text = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {detected_text}")
            target_object = detected_text  # Store user-specified object
            speak(f"Detecting only {target_object}")
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError:
            print("Error connecting to speech recognition service")

# Start voice recognition in a separate thread
voice_thread = threading.Thread(target=recognize_speech, daemon=True)
voice_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    detected_objects = results.pandas().xyxy[0]  # Pandas DataFrame with detections

    current_detections = set()  # Store objects detected in this frame

    for _, row in detected_objects.iterrows():
        label = row['name']
        confidence = row['confidence']

        # If the user specified an object, only detect that one
        if target_object and label != target_object:
            continue

        if confidence > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store detected objects for audio feedback
            current_detections.add(label)

    # Announce detected objects, avoiding redundant speech
    new_detections = current_detections - previous_detections
    if new_detections:
        speak(", ".join(new_detections) + " detected")

    # Update previous detections
    previous_detections = current_detections

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
