import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path("/Users/almansurkakimov/Downloads/open")
VIDEOS_DIR = BASE_DIR / "videos"
LABELS_FILE = BASE_DIR / "open_labels.xlsx"

def run_audit():
    data = []
    video_files = list(VIDEOS_DIR.glob("*.mov"))
    
    if not video_files:
        print("No .mov files found in", VIDEOS_DIR)
        return

    for video_file in tqdm(video_files, desc="Profiling Videos"):
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_file)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        out = result.stdout.strip().split('\n')
        
        if len(out) == 5:
            codec, w, h, fps, duration = out
            data.append({
                "filename": video_file.name,
                "codec": codec,
                "width": int(w),
                "height": int(h),
                "fps": fps,
                "duration_sec": float(duration)
            })
        else:
            print(f"Failed to parse {video_file.name}: {out}")

    df_videos = pd.DataFrame(data)
    df_videos.to_csv("video_audit.csv", index=False)
    print(f"\nAudited {len(df_videos)} videos.")
    print("Codec breakdown:\n", df_videos['codec'].value_counts())
    print("Resolution breakdown:\n", df_videos.groupby(['width', 'height']).size())
    
    if LABELS_FILE.exists():
        df_labels = pd.read_excel(LABELS_FILE)
        print("\n=== Labels File Info ===")
        print("Columns:", df_labels.columns.tolist())
        print(df_labels.head(3))
    else:
        print("\nWARNING: open_labels.xlsx not found at", LABELS_FILE)

if __name__ == "__main__":
    run_audit()