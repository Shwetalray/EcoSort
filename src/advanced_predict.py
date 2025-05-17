import cv2
import torch
import numpy as np
import time
import mediapipe as mp
import os
import json
import serial
import time
import cv2

def draw_blinking_border(frame, color=(0, 0, 255), thickness=20):
    """Draws a blinking red border."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
    return frame

def show_frozen_frame(label, frame):
    # No blinking or text here, just a clean frozen frame with bold border
    bordered_frame = draw_blinking_border(frame.copy())  # Border remains red
    small_frame = cv2.resize(bordered_frame, (0, 0), fx=0.5, fy=0.5)
    color = (0, 255, 0) if label == "Paper Detected" else (255, 0, 0)
    cv2.putText(small_frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    cv2.imshow("Detected Frame", small_frame)
    cv2.waitKey(2000)
    cv2.destroyWindow("Detected Frame")

# Initialize serial and camera
ser = serial.Serial('COM3', 9600)
time.sleep(3)
camera = cv2.VideoCapture(0)

print("✅ System Ready. Waiting for ultrasonic trigger...")

detect_paper_next = True  # Toggle variable

while True:
    # Wait for ultrasonic trigger
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line == "CAM_START":
                print("[INFO] Ultrasonic triggered. Waiting 3 seconds before opening camera...")
                time.sleep(3)
                print("[INFO] Starting camera preview...")
                break

    start_time = time.time()
    freeze_frame = None
    blink = True  # For blinking effect

    # Show live feed with blinking "Detecting........." and red border
    while time.time() - start_time < 3.8:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to capture from camera.")
            break

        display_frame = frame.copy()

        if blink:
            # Blinking red border
            display_frame = draw_blinking_border(display_frame, color=(0, 0, 255), thickness=20)

            # Red text in top-right corner (large and bold)
            text = "Detecting........."
            font_scale = 1.8
            text_thickness = 4
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            text_x = display_frame.shape[1] - text_size[0] - 30
            text_y = 60
            cv2.putText(display_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 255), text_thickness)

        cv2.imshow("Live Feed (Press 'q' to Quit)", display_frame)

        if cv2.waitKey(250) & 0xFF == ord('q'):
            print("[EXIT] 'q' pressed. Exiting...")
            camera.release()
            ser.close()
            cv2.destroyAllWindows()
            exit()

        freeze_frame = frame.copy()
        blink = not blink  # Toggle blinking state

    # Prepare and show frozen frame with result
    label = "Paper Detected" if detect_paper_next else "Plastic Detected"
    command = b"ROTATE_PAPER\n" if detect_paper_next else b"ROTATE_PLASTIC\n"

    print(f"[INFO] {label} → Frame frozen. Displaying result...")
    show_frozen_frame(label, freeze_frame)

    print("[WAIT] Waiting 2 seconds before sending rotate command...")
    time.sleep(2.2)

    print(f"[ACTION] Sending command → {command.decode().strip()}")
    ser.write(command)

    detect_paper_next = not detect_paper_next  # Toggle next type
    cv2.destroyWindow("Live Feed (Press 'q' to Quit)")
    print("[INFO] Waiting for next ultrasonic trigger...\n")
