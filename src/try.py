import torch
import numpy as np
import time
import mediapipe as mp
import os
import json
import cv2
import serial
import time
import winsound 

def play_beep(frequency=1000, duration=200):
    winsound.Beep(frequency, duration)

def draw_blinking_border(frame, blink_state=True, color=(0, 0, 255), thickness=20):
    if blink_state:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
    return frame

def show_scanning_effect(frame):
    scan_frame = frame.copy()
    h, w = scan_frame.shape[:2]
    step = 10
    for i in range(0, h, step):
        overlay = scan_frame.copy()
        cv2.line(overlay, (0, i), (w, i), (0, 255, 0), 2)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, scan_frame, 1 - alpha, 0, scan_frame)
    return scan_frame

def show_frozen_result(label, frame):
    color = (0, 255, 0) if label == "Paper Detected" else (255, 0, 0)
    result_frame = frame.copy()
    h, w = result_frame.shape[:2]
    cv2.rectangle(result_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 20)
    cv2.putText(result_frame, label, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)
    cv2.imshow("Detected Frame", cv2.resize(result_frame, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(2000)
    cv2.destroyWindow("Detected Frame")

# Serial & camera setup
ser = serial.Serial('COM3', 9600)
time.sleep(3)
camera = cv2.VideoCapture(0)
print("✅ System Ready. Waiting for ultrasonic trigger...")

while True:
    # Wait for ultrasonic trigger
    while True:
        if ser.in_waiting > 0 and ser.readline().decode().strip() == "CAM_START":
            print("[INFO] Ultrasonic triggered. Initializing camera...")
            time.sleep(3)
            break

    blink = True
    freeze_frame = None

    print("[INFO] Camera preview active. Press 'q' to quit.")

    # Show live feed with blinking red border and text
    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Camera failure.")
            break

        display_frame = frame.copy()
        display_frame = draw_blinking_border(display_frame, blink)

        text = "Detecting........."
        font_scale, thickness = 1.8, 4
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.putText(display_frame, text,
                    (display_frame.shape[1] - text_size[0] - 30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        cv2.imshow("Live Feed", display_frame)
        key = cv2.waitKey(250) & 0xFF

        if key == ord('q'):
            print("[EXIT] Quitting.")
            camera.release()
            ser.close()
            cv2.destroyAllWindows()
            exit()
        elif key in [ord('g'), ord('h')]:
            label = "Paper Detected" if key == ord('g') else "Plastic Detected"
            command = b"ROTATE_PAPER\n" if key == ord('g') else b"ROTATE_PLASTIC\n"
            freeze_frame = frame.copy()
            play_beep(1000, 200)  # Beep on key press
            break

        blink = not blink

    # Show scanning effect for 2 seconds
    scan_start = time.time()
    while time.time() - scan_start < 2:
        scanned_frame = show_scanning_effect(freeze_frame)
        cv2.imshow("Scanning...", cv2.resize(scanned_frame, (0, 0), fx=0.5, fy=0.5))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            ser.close()
            cv2.destroyAllWindows()
            exit()
    cv2.destroyWindow("Scanning...")

    # Show final result for 2 seconds
    print(f"[INFO] {label}")
    show_frozen_result(label, freeze_frame)

    print(f"[ACTION] Sending command: {command.decode().strip()}")
    ser.write(command)

    cv2.destroyWindow("Live Feed")
    print("[INFO] Waiting for next ultrasonic trigger...\n")
