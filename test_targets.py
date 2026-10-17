import time
import cv2
import gc
import torch
from pathlib import Path
from engine import CashSecurityEngine

TARGET_NUMBERS = ["24", "28", "40"]
VIDEOS_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")

def run_targets():
    print(f"🚀 Verifying fixes on False Negatives {TARGET_NUMBERS} (Require Violations)...")
    engine = CashSecurityEngine()
    
    for target in TARGET_NUMBERS:
        matched_files = [f for f in VIDEOS_DIR.glob("*.mov") if f.name.endswith(f"-{target}.mov")]
        if not matched_files: continue
            
        video_file = matched_files[0]
        print(f"\n🎥 Analyzing {video_file.name}...")
        
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 30
        
        frame_idx = 0
        seconds_processed = 0
        video_violations = []
        
        for rule_key in engine.violation_streaks.keys():
            engine.violation_streaks[rule_key] = {"count": 0, "first_seen_sec": 0, "reported": False}
            
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_sec = seconds_processed + 1
                
                _, frame_violations = engine.analyze_frame(frame_rgb, current_sec)
                
                if frame_violations:
                    for v in frame_violations:
                        print(f"   ⚠️ [Sec {v['timecode_sec']}] Flagged: {v['type']}")
                    video_violations.extend(frame_violations)
                    # Early stop
                    break
                    
                seconds_processed += 1
            frame_idx += 1
            
        cap.release()
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
        duration = time.time() - start_time
        if video_violations:
            print(f"   ✅ SUCCESS: Violation caught! (Processed in {duration:.1f}s)")
        else:
            print(f"   ❌ FAILED: Engine missed the violation. (Processed in {duration:.1f}s)")

if __name__ == "__main__":
    run_targets()