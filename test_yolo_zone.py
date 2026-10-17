import cv2
import time
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH = "/Users/almansurkakimov/Downloads/open/videos/X-1.mov"
OUTPUT_IMAGE = "zone_debug.jpg"

def test_yolo_and_zoning():
    print("Loading YOLOv8n on MPS...")
    # Initialize YOLO and force it to use the M1 GPU
    model = YOLO("yolov8n.pt")
    
    # Open the video and skip to the 1st second (approx frame 30)
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return
        
    height, width, _ = frame.shape
    print(f"Frame resolution: {width}x{height}")

    # Proposing a horizontal split: Top 60% is Cashier Zone, Bottom 40% is Client Zone
    # You will look at the output image to see if this line needs to move up or down.
    split_y = int(height * 0.6)

    # Run inference and measure time
    start_time = time.time()
    results = model.predict(frame, device="mps", classes=[0], verbose=False) # class 0 is 'person'
    inference_time = (time.time() - start_time) * 1000

    print(f"✅ MPS Inference Time: {inference_time:.2f} ms")
    print(f"✅ People detected: {len(results[0].boxes)}")

    # Annotate the frame
    annotated_frame = results[0].plot()
    
    # Draw the zoning line (Green line)
    cv2.line(annotated_frame, (0, split_y), (width, split_y), (0, 255, 0), 3)
    cv2.putText(annotated_frame, "CASHIER ZONE (SECURE)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "CLIENT ZONE", (10, split_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(OUTPUT_IMAGE, annotated_frame)
    print(f"Saved debug image to {OUTPUT_IMAGE}. Please open it and check the green line.")

if __name__ == "__main__":
    test_yolo_and_zoning()