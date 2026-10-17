import time
import cv2
from engine import CashSecurityEngine

VIDEO_PATH = "/Users/almansurkakimov/Downloads/open/videos/X-1.mov"

def run_benchmark():
    print("=== M1 Benchmarking (Full Video) ===")
    
    # Init engine (loads on MPS by default)
    init_start = time.time()
    engine = CashSecurityEngine()
    print(f"Model Load Time: {time.time() - init_start:.2f}s")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    frame_interval = int(fps) if fps > 0 else 30
    
    print(f"\nVideo Length: {video_duration:.2f} seconds")
    print("Processing at 1 Frame Per Second (1 FPS)...")
    
    frame_idx = 0
    processed_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract 1 frame per second
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run the hybrid engine (YOLO + Conditional VLM)
            engine.analyze_frame(frame_rgb)
            
            processed_count += 1
            if processed_count % 5 == 0:
                print(f"Processed {processed_count} frames...")
                
        frame_idx += 1
        
    cap.release()
    
    total_time = time.time() - start_time
    print("\n=== Benchmark Results ===")
    print(f"Frames analyzed by ML: {processed_count}")
    print(f"Target limit: < 300 seconds (5 minutes)")
    print(f"Actual Processing Time: {total_time:.2f} seconds")
    
    if total_time < 300:
        print("✅ VERDICT: PASS (Fully compliant with Hackathon constraints)")
    else:
        print("❌ VERDICT: FAIL (Needs optimization)")

if __name__ == "__main__":
    run_benchmark()