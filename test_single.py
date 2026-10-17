import cv2
import time
from PIL import Image
from engine import CashSecurityEngine
from pathlib import Path

VIDEOS_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
matched_files = list(VIDEOS_DIR.glob("*15.mov"))

def run_debug():
    if not matched_files: return
    actual_video_path = str(matched_files[0])
    print(f"🔍 Debugging VLM perception on {matched_files[0].name}...")
    
    engine = CashSecurityEngine()
    cap = cv2.VideoCapture(actual_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) if fps > 0 else 30
    
    frame_idx = 0
    seconds_processed = 0
    
    while cap.isOpened() and seconds_processed <= 20:
        ret, frame = cap.read()
        if not ret: break
            
        if frame_idx % frame_interval == 0:
            current_sec = seconds_processed + 1
            
            # Only debug the critical window
            if 13 <= current_sec <= 18:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                print(f"\n--- Second {current_sec} ---")
                results = engine.yolo.predict(frame_rgb, classes=[0], device=engine.device, verbose=False)
                print(f"YOLO People Count: {len(results[0].boxes)}")
                
                enc_img = engine.vlm.encode_image(pil_img)
                
                # Question 1: Sitting?
                ans1 = engine.vlm.answer_question(enc_img, "Is the person sitting down in the chair at the desk? Answer strictly yes or no.", engine.tokenizer)
                print(f"Q: Sitting? -> {ans1.strip()}")
                
                # Question 2: Open-ended bag check
                ans2 = engine.vlm.answer_question(enc_img, "What objects are lying on the wooden desk?", engine.tokenizer)
                print(f"Q: What objects? -> {ans2.strip()}")
                
                # Question 3: Simplified bag check
                ans3 = engine.vlm.answer_question(enc_img, "Is there a bag or pouch lying on the desk? Answer strictly yes or no.", engine.tokenizer)
                print(f"Q: Bag on desk? -> {ans3.strip()}")
                
            seconds_processed += 1
        frame_idx += 1
        
    cap.release()
    print("\nDebug finished.")

if __name__ == "__main__":
    run_debug()