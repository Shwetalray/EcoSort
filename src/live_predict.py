import cv2
import torch
import numpy as np
from PIL import Image
from train_model import WasteClassifier
from pathlib import Path
import time

class SmartWastePredictor:
    def __init__(self):
        self.classifier = WasteClassifier()
        self.model = self.classifier.setup_model()
        
        # Load the trained model
        model_path = Path("dataset/best_model.pth")
        if not model_path.exists():
            raise FileNotFoundError("Trained model not found! Please train the model first.")
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam!")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Parameters for object detection
        self.min_object_size = 1000  # Minimum contour area to be considered an object
        self.frame_lock_time = 2  # Time to wait before capturing next frame
        self.confidence_threshold = 0.7  # Minimum confidence for prediction
        
    def detect_object(self, frame):
        """Detect if there's a significant object in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_object_size:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                return True, (x, y, w, h)
        
        return False, None
    
    def preprocess_frame(self, frame, bbox=None):
        """Preprocess frame for model prediction"""
        if bbox:
            x, y, w, h = bbox
            # Add padding around the object
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            frame = frame[y:y+h, x:x+w]
        
        # Convert to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Remove background if enabled
        if self.classifier.remove_background:
            temp_dataset = self.classifier.WasteDataset(self.classifier.base_path, "val_annotations.json")
            image = temp_dataset.remove_background_from_image(image, Path("temp.jpg"))
        
        # Apply transformations
        image = self.classifier.val_transform(image)
        return image.unsqueeze(0).to(self.classifier.device)
    
    def predict_frame(self, frame, bbox=None):
        """Make prediction on a frame"""
        processed_frame = self.preprocess_frame(frame, bbox)
        
        with torch.no_grad():
            outputs = self.model(processed_frame)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() >= self.confidence_threshold:
                return self.classifier.categories[predicted.item()], confidence.item()
            else:
                return "Uncertain", confidence.item()
    
    def run(self):
        print("Starting smart waste detection... Press 'q' to quit")
        
        last_detection_time = 0
        locked_frame = None
        locked_bbox = None
        prediction = None
        confidence = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame")
                break
            
            current_time = time.time()
            display_frame = frame.copy()
            
            # Only process new frame if not locked or lock time expired
            if locked_frame is None or (current_time - last_detection_time) > self.frame_lock_time:
                object_detected, bbox = self.detect_object(frame)
                
                if object_detected:
                    locked_frame = frame.copy()
                    locked_bbox = bbox
                    last_detection_time = current_time
                    prediction, confidence = self.predict_frame(locked_frame, locked_bbox)
            
            # Draw on display frame
            if locked_frame is not None and locked_bbox is not None:
                x, y, w, h = locked_bbox
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw prediction
                if prediction:
                    text = f"{prediction} ({confidence:.2%})"
                    cv2.putText(display_frame, text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                # Draw lock indicator
                time_left = max(0, self.frame_lock_time - (current_time - last_detection_time))
                if time_left > 0:
                    cv2.putText(display_frame, f"Locked: {time_left:.1f}s", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Smart Waste Classification', display_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        predictor = SmartWastePredictor()
        predictor.run()
    except KeyboardInterrupt:
        print("\nStopping prediction...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 