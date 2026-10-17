import time
import cv2
import pandas as pd
from pathlib import Path
from engine import CashSecurityEngine

VIDEOS_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")

def run_batch():
    engine = CashSecurityEngine()
    
    # Grab the first 3 videos for a quick smoke test
    video_files = list(VIDEOS_DIR.glob("*.mov"))[:3]
    
    results = []
    
    for video_file in video_files:
        print(f"\n🎥 Analyzing {video_file.name}...")
        start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 30
        
        frame_idx = 0
        seconds_processed = 0
        video_violations = []
        
        # Reset the temporal smoother for each new video!
        engine.violation_streaks = {
            "unauthorized_person": {"count": 0, "first_seen_sec": 0, "reported": False},
            "unattended_open_safe": {"count": 0, "first_seen_sec": 0, "reported": False}
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_sec = seconds_processed + 1
                
                _, frame_violations = engine.analyze_frame(frame_rgb, current_sec)
                if frame_violations:
                    video_violations.extend(frame_violations)
                    
                seconds_processed += 1
            frame_idx += 1
            
        cap.release()
        duration = time.time() - start_time
        
        verdict = "нарушение" if len(video_violations) > 0 else "без нарушения"
        print(f"✅ Finished in {duration:.2f}s | Verdict: {verdict} | Violations found: {len(video_violations)}")
        
        results.append({
            "filename": video_file.name,
            "processing_time_sec": round(duration, 2),
            "verdict": verdict,
            "violations": video_violations
        })

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv("batch_summary.csv", index=False)
    print("\nBatch test complete! Summary saved to batch_summary.csv.")

if __name__ == "__main__":
    run_batch()