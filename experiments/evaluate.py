import time
import cv2
import pandas as pd
import gc
import torch
from pathlib import Path
from engine import CashSecurityEngine

BASE_DIR = Path("/Users/almansurkakimov/Downloads/open")
VIDEOS_DIR = BASE_DIR / "videos"
LABELS_FILE = BASE_DIR / "open_labels.xlsx"

def normalize_name(name):
    # Hackathon data often mixes Cyrillic 'Х' and Latin 'X'
    return str(name).replace('.mov', '').replace('Х', 'X').replace('х', 'X').strip()

def run_evaluation():
    print("Loading ground truth labels...")
    df_labels = pd.read_excel(LABELS_FILE)
    df_labels.columns = ['filename', 'timestamp', 'label']
    df_labels['norm_filename'] = df_labels['filename'].apply(normalize_name)
    
    engine = CashSecurityEngine()
    video_files = list(VIDEOS_DIR.glob("*.mov"))
    
    print(f"\n🚀 Starting full evaluation on {len(video_files)} videos.")
    print("⚠️ WARNING: This may take 30-45 minutes. Let it run!\n")
    
    results = []
    
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Analyzing {video_file.name}...")
        start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 30
        
        frame_idx = 0
        seconds_processed = 0
        video_violations = []
        
        # Reset state tracker for new video (NOW INCLUDES RULE 3)
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
        
        # Force garbage collection to prevent M1 Unified Memory bloat
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        verdict = "нарушение" if len(video_violations) > 0 else "без нарушения"
        duration = time.time() - start_time
        print(f"   -> {verdict} (Found {len(video_violations)} violations) in {duration:.1f}s")
        
        results.append({
            "norm_filename": normalize_name(video_file.name),
            "predicted": verdict,
            "violations_count": len(video_violations)
        })
        
    # Merge and calculate metrics
    df_results = pd.DataFrame(results)
    df_eval = pd.merge(df_labels, df_results, on="norm_filename", how="inner")
    
    # Check if predicted matches the ground truth label
    df_eval['correct'] = (df_eval['label'].str.strip().str.lower() == df_eval['predicted'].str.lower())
    
    accuracy = df_eval['correct'].mean() * 100
    print("\n" + "="*40)
    print(f"✅ FULL DATASET EVALUATION COMPLETE")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print("="*40)
    
    # Save the detailed report so we can see exactly where it failed
    df_eval.to_csv("full_evaluation_report.csv", index=False)
    print("Saved 'full_evaluation_report.csv'.")

if __name__ == "__main__":
    run_evaluation()