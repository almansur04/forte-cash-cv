import pandas as pd, cv2, subprocess, json, re
from pathlib import Path
from PIL import Image, ImageDraw

VIDEO_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
LABELS_FILE = Path("/Users/almansurkakimov/Downloads/open/open_labels.xlsx")
BENCH_CSV = Path("baseline/benchmark_full_results.csv")
OUT_IMG = Path("baseline/diag_fns_fps.jpg")
FPS = 30.0

def norm(n): return str(n).replace('.mov','').replace('Х','X').replace('х','X').strip()
def parse_ts(s):
    if pd.isna(s): return 5
    s = str(s).lower()
    m = re.search(r"(\d+)\s*минут\w*\s*(\d+)", s)
    if m: return int(m.group(1))*60 + int(m.group(2))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 5

df_bench = pd.read_csv(BENCH_CSV)
df_labels = pd.read_excel(LABELS_FILE); df_labels.columns = ['filename','ts_raw','label']
df_labels['norm'] = df_labels['filename'].apply(norm)

fps_list = df_bench[(df_bench['true_label']=='без нарушения') & (df_bench['predicted']=='нарушение')]['video'].tolist()
fns_list = df_bench[(df_bench['true_label']=='нарушение') & (df_bench['predicted']=='без нарушения')]['video'].tolist()

def grab(v_name, t_sec):
    f = next((v for v in VIDEO_DIR.glob("*.mov") if norm(v.stem)==norm(v_name)), None)
    if not f: return None, f"Missing {v_name}"
    cap = cv2.VideoCapture(str(f))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * FPS))
    ok, frame = cap.read(); cap.release()
    if not ok: return None, f"Read fail {v_name}"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"{v_name} t={t_sec}s"

rows = []
for v in fns_list[:10]:
    lbl = df_labels[df_labels['norm']==norm(v)].iloc[0]
    t = parse_ts(lbl['ts_raw'])
    img, title = grab(v, t)
    if img is not None: rows.append((Image.fromarray(img), f"FN: {title}"))

for v in fps_list[:10]:
    img, title = grab(v, 8)
    if img is not None: rows.append((Image.fromarray(img), f"FP: {title}"))

if rows:
    w0, h0 = rows[0][0].size
    scale = 250 / h0
    tw, th = int(w0*scale), int(h0*scale)
    cols = 5
    r = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", (cols*tw + 10*(cols+1), r*(th+30) + 10*(r+1)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, title) in enumerate(rows):
        rr, cc = divmod(i, cols)
        x = 10 + cc*(tw+10); y = 10 + rr*(th+30+10)
        canvas.paste(img.resize((tw, th)), (x, y+20))
        draw.text((x, y), title, fill="black")
    canvas.save(OUT_IMG, "JPEG", quality=90)
    print(f"Saved {OUT_IMG}")
