import cv2
import time
from engine import CashSecurityEngine
from pathlib import Path

# Search the directory for ANY file ending in "11.mov"
VIDEOS_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
matched_files = list(VIDEOS_DIR.glob("*11.mov"))

def run_test():
    if not matched_files:
        print(f"Error: Could not find any video ending in 11.mov in {VIDEOS_DIR}")
        return
        
    actual_video_path = str(matched_files[0])
    print(f"Testing new Rule 3 on {matched_files[0].name}...")
    
    engine = CashSecurityEngine()
    
    cap = cv2.VideoCapture(actual_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) if fps > 0 else 30
    
    frame_idx = 0
    seconds_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_sec = seconds_processed + 1
            
            p_count, frame_violations = engine.analyze_frame(frame_rgb, current_sec)
            
            if frame_violations:
                print(f"⚠️ [Sec {current_sec}] VIOLATION DETECTED: {frame_violations}")
                
            seconds_processed += 1
        frame_idx += 1
        
    cap.release()
    print("Test finished.")

if __name__ == "__main__":
    run_test()