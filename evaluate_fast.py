import time
import cv2
import pandas as pd
import gc
import torch
from pathlib import Path
from engine import CashSecurityEngine

BASE_DIR = Path("/Users/almansurkakimov/Downloads/open")
VIDEOS_DIR = BASE_DIR / "videos"

def run_fast_evaluation():
    try:
        df_old = pd.read_csv("full_evaluation_report.csv")
    except FileNotFoundError:
        print("Error: full_evaluation_report.csv not found.")
        return
        
    # Get ONLY the videos that were marked as incorrect in the last run
    df_failed = df_old[df_old['correct'] == False].copy()
    
    if df_failed.empty:
        print("No failed videos found in the previous report!")
        return

    print(f"🚀 Fast Evaluation: Testing ONLY the {len(df_failed)} previously failed videos...")
    
    engine = CashSecurityEngine()
    fixed_count = 0
    
    for index, row in df_failed.iterrows():
        # Handle Cyrillic/Latin naming dynamically
        possible_files = list(VIDEOS_DIR.glob(f"*{row['norm_filename'].replace('X', '')}.mov"))
        if not possible_files:
            print(f"Skipping {row['norm_filename']}: File not found.")
            continue
            
        video_file = possible_files[0]
        print(f"\nAnalyzing {video_file.name} (True Label: {row['label']})...")
        
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 30
        
        frame_idx = 0
        seconds_processed = 0
        video_violations = []
        
        # Reset state tracker
        engine.violation_streaks = {
            "unauthorized_person": {"count": 0, "first_seen_sec": 0, "reported": False},
            "unattended_open_safe": {"count": 0, "first_seen_sec": 0, "reported": False},
            "unattended_workstation": {"count": 0, "first_seen_sec": 0, "reported": False}
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_sec = seconds_processed + 1
                _, frame_violations = engine.analyze_frame(frame_rgb, current_sec)
                
                if frame_violations:
                    video_violations.extend(frame_violations)
                    
                seconds_processed += 1
            frame_idx += 1
            
        cap.release()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        verdict = "нарушение" if len(video_violations) > 0 else "без нарушения"
        duration = time.time() - start_time
        
        is_now_correct = (verdict.lower() == row['label'].strip().lower())
        
        if is_now_correct:
            print(f"   ✅ FIXED! Engine now correctly predicts '{verdict}' (in {duration:.1f}s)")
            fixed_count += 1
        else:
            print(f"   ❌ Still incorrect. Engine predicted '{verdict}' (in {duration:.1f}s)")

    # Calculate projected accuracy
    total_videos = len(df_old)
    previously_correct = len(df_old[df_old['correct'] == True])
    projected_accuracy = ((previously_correct + fixed_count) / total_videos) * 100
    
    print("\n" + "="*45)
    print(f"🎯 FAST EVALUATION COMPLETE")
    print(f"Videos fixed by Rule 3: {fixed_count} out of {len(df_failed)}")
    print(f"📈 Projected New Accuracy: {projected_accuracy:.2f}% (Up from 52.38%)")
    print("="*45)

if __name__ == "__main__":
    run_fast_evaluation()