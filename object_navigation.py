import cv2
import torch
import pyttsx3
import numpy as np
import threading
import speech_recognition as sr
import datetime
import google.generativeai as genai
import os
import pickle
import time
import queue
from PIL import Image
import pytesseract

# Initialize TTS engine
engine = pyttsx3.init()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam
cap = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH = int(cap.get(3))

# Variables for speech recognition
stop_flag = False
target_object = None
previous_detections = set()
last_speech_time = 0
speech_cooldown = 3
speech_queue = queue.Queue()


# Function to speak
def speak(text):
    speech_queue.put(text)


def speak_thread():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()


speech_thread = threading.Thread(target=speak_thread, daemon=True)
speech_thread.start()


# Voice command listener
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  # Added timeout to avoid freezing
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {command}")
            return command
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
        except sr.WaitTimeoutError:
            speak("No command detected.")
    return None


# Object detection with navigation
def get_navigation_direction(x1, x2, width):
    center_x = (x1 + x2) // 2
    left_threshold = width // 3
    right_threshold = 2 * (width // 3)
    if center_x < left_threshold:
        return "Move left"
    elif center_x > right_threshold:
        return "Move right"
    else:
        return "Move forward"


def estimate_distance(obj_width):
    if obj_width > 300:
        return "Very close"
    elif obj_width > 200:
        return "Near"
    elif obj_width > 100:
        return "Medium distance"
    else:
        return "Far"


def object_detection():
    global last_speech_time, target_object
    cap = cv2.VideoCapture(1)  # Ensure webcam is opened properly

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detected_objects = results.pandas().xyxy[0]  # Get detections as a DataFrame

        for _, row in detected_objects.iterrows():
            label = row['name']
            confidence = row['confidence']

            # Check if detected object matches the target object
            if target_object and target_object not in label.lower():
                continue  # Skip detection if it's not the requested object

            if confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                direction = get_navigation_direction(x1, x2, FRAME_WIDTH)
                distance = estimate_distance(x2 - x1)

                current_time = time.time()
                if (current_time - last_speech_time) > speech_cooldown:
                    speak(f"{label} detected. {direction}. It is {distance}.")
                    last_speech_time = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection and Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def recognize_faces():
    global stop_flag
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read("face_model.xml")
    except cv2.error:
        print("Face recognizer model not found. Ensure face_model.xml is available.")
        return

    try:
        with open("labels.pkl", "rb") as f:
            label_map = pickle.load(f)
    except FileNotFoundError:
        print("labels.pkl not found. Proceeding without known faces.")
        label_map = {}

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label_id, confidence = face_recognizer.predict(face_img)
            name = label_map.get(label_id, "Unknown") if confidence < 58 else "Unknown"
            speak(f"{name} is recognized" if name != "Unknown" else "Unknown person detected")
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    global target_object
    while True:
        command = listen_command()
        if command:
            print(f"User command: {command}")  # Debugging log
            if "time" in command:
                now_time = datetime.datetime.now().strftime("%I:%M %p")
                speak(f"The time is {now_time}")
            elif "date" in command:
                date = datetime.datetime.now().strftime("%B %d, %Y")
                speak(f"Today's date is {date}")
            elif "recognize faces" in command:
                recognize_faces()
            elif "object" in command:
                speak("What object do you want to detect?")
                time.sleep(1)  # Ensure proper microphone activation
                target_object = listen_command()

                # Debugging logs
                print(f"Target object recognized: {target_object}")

                if target_object:
                    speak(f"Detecting {target_object}")
                    object_detection()
                else:
                    speak("Sorry, I couldn't recognize the object.")
            elif "exit" in command:
                speak("Goodbye!")
                break


if __name__ == "__main__":
    main()
    speech_queue.put(None)
    speech_thread.join()
