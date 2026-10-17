import time, json, psutil, os, re, gc, torch
import pandas as pd
from pathlib import Path
from engine import CashSecurityEngine

VIDEO_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
LABELS_FILE = Path("/Users/almansurkakimov/Downloads/open/open_labels.xlsx")
OUT_CSV = Path("baseline/benchmark_full_results.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

def normalize_name(name):
    return str(name).replace('.mov', '').replace('Х', 'X').replace('х', 'X').strip()

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def main():
    print("Loading labels...")
    df_labels = pd.read_excel(LABELS_FILE)
    df_labels.columns = ['filename', 'timestamp', 'label']
    df_labels['norm_filename'] = df_labels['filename'].apply(normalize_name)
    
    print("Initializing engine (YOLOv8n on MPS)...")
    eng = CashSecurityEngine(rules_csv="violation_rules_verified.csv")
    
    video_files = sorted(list(VIDEO_DIR.glob("*.mov")))
    print(f"Found {len(video_files)} videos. Starting full benchmark...\n")
    
    results = []
    total_start = time.time()
    
    for i, vf in enumerate(video_files, 1):
        norm_name = normalize_name(vf.name)
        
        # Match with labels
        lbl_row = df_labels[df_labels['norm_filename'] == norm_name]
        if lbl_row.empty:
            print(f"[{i}/{len(video_files)}] {vf.name} - WARNING: no label found, skipping.")
            continue
        true_label = lbl_row.iloc[0]['label'].strip().lower()
        
        # Benchmark
        t0 = time.time()
        r = eng.analyze_video(vf, sample_fps=2, evidence_dir=None)
        elapsed = time.time() - t0
        mem_mb = get_memory_mb()
        
        pred_label = r['verdict'].strip().lower()
        is_correct = (true_label == pred_label)
        
        results.append({
            "video": vf.name,
            "true_label": true_label,
            "predicted": pred_label,
            "correct": is_correct,
            "time_sec": round(elapsed, 2),
            "mem_mb": round(mem_mb, 1),
            "violations_count": len(r['violations']),
            "violation_types": ",".join([v['type'] for v in r['violations']])
        })
        
        print(f"[{i}/{len(video_files)}] {vf.name:12s} | True: {true_label:15s} | Pred: {pred_label:15s} | Time: {elapsed:5.2f}s | Mem: {mem_mb:5.1f}MB | {'✅' if is_correct else '❌'}")
        
        # Cleanup
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    total_elapsed = time.time() - total_start
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_CSV, index=False)
    
    # Metrics
    tp = len(df_res[(df_res['true_label'] == 'нарушение') & (df_res['predicted'] == 'нарушение')])
    tn = len(df_res[(df_res['true_label'] == 'без нарушения') & (df_res['predicted'] == 'без нарушения')])
    fp = len(df_res[(df_res['true_label'] == 'без нарушения') & (df_res['predicted'] == 'нарушение')])
    fn = len(df_res[(df_res['true_label'] == 'нарушение') & (df_res['predicted'] == 'без нарушения')])
    
    accuracy = (tp + tn) / len(df_res) * 100 if len(df_res) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("🎯 FINAL BENCHMARK RESULTS (M1 MPS, 2 FPS)")
    print("="*50)
    print(f"Videos Processed: {len(df_res)}")
    print(f"Total Time:       {total_elapsed:.2f}s")
    print(f"Avg Time/Video:   {df_res['time_sec'].mean():.2f}s  (Max: {df_res['time_sec'].max():.2f}s)")
    print(f"Peak RAM:         {df_res['mem_mb'].max():.1f} MB")
    print("-" * 50)
    print(f"Accuracy:         {accuracy:.2f}%  ({(tp+tn)}/{len(df_res)})")
    print(f"Precision:        {precision:.2f}%  ({tp} TP / {tp+fp} total positives)")
    print(f"Recall:           {recall:.2f}%  ({tp} TP / {tp+fn} actual violations)")
    print(f"F1-Score:         {f1:.2f}%")
    print("-" * 50)
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}  <-- MUST BE MINIMIZED")
    print("="*50)
    
    if fp > 0:
        print("\nFalse Positive Videos (flagged but clean):")
        print(df_res[(df_res['true_label'] == 'без нарушения') & (df_res['predicted'] == 'нарушение')][['video', 'time_sec', 'violation_types']].to_string(index=False))
    if fn > 0:
        print("\nFalse Negative Videos (missed violations):")
        print(df_res[(df_res['true_label'] == 'нарушение') & (df_res['predicted'] == 'без нарушения')][['video', 'time_sec']].to_string(index=False))
        
    print(f"\nDetailed CSV saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()
