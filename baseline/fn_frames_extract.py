"""Extract 1 frame per FN at the labeled timestamp + Moondream open-ended describe.
   Assembling the montage is a separate PIL step (open in Preview)."""
import cv2, re, json, subprocess
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VIDEO_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
LABELS    = Path("/Users/almansurkakimov/Downloads/open/open_labels.xlsx")
OUT_DIR   = Path("baseline/fn_frames"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON  = Path("baseline/fn_frames.json")

FN = ["Х-11","Х-13","Х-15","Х-20","Х-22","Х-24","Х-26","Х-28","Х-32","Х-40","Х-44","Х-46"]

def parse_ts(s):
    if pd.isna(s): return None
    s = str(s).lower()
    m = re.search(r"(\d+)\s*минут\w*\s*(\d+)", s)
    if m: return int(m.group(1))*60 + int(m.group(2))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def _norm(x): return str(x).replace("Х","X").replace("х","X").strip()

def find_file(name):
    n = _norm(name)
    for f in VIDEO_DIR.glob("*.mov"):
        if _norm(f.stem) == n: return f
    return None

def ffprobe_fps(video):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=r_frame_rate","-of","csv=p=0", str(video)],
            capture_output=True, text=True, timeout=10)
        num, den = r.stdout.strip().split("/")
        return float(num)/float(den)
    except Exception:
        return 30.0

def ffprobe_duration(video):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0", str(video)],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip())
    except Exception:
        return None

def grab_frame_at(video, t_sec):
    cap = cv2.VideoCapture(str(video))
    fps = ffprobe_fps(video)
    dur = ffprobe_duration(video)
    if dur is not None and t_sec > dur: t_sec = max(1, int(dur) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok: return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def make_montage(paths, labels, out_path, cols=4):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w0, h0 = imgs[0].size
    scale = 300 / h0
    tw, th = int(w0*scale), int(h0*scale)
    rows = (len(imgs) + cols - 1) // cols
    margin = 40
    canvas = Image.new("RGB", (cols*tw + 20*(cols+1), rows*(th+margin) + 20*(rows+1)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        x = 20 + c*(tw+20); y = 20 + r*(th+margin+20)
        canvas.paste(img.resize((tw, th)), (x, y+margin))
        draw.text((x, y), lab, fill="black")
    canvas.save(out_path, "JPEG", quality=88)
    return out_path

def main():
    print("Loading Moondream2 on MPS ...", flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "vikhyatk/moondream2"; revision = "2024-08-26"
    vlm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision).to(device)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    df = pd.read_excel(LABELS); df.columns = ["filename","ts_raw","label"]
    df["ts_sec"] = df["ts_raw"].apply(parse_ts)

    results = []
    paths_for_montage = []
    labels_for_montage = []

    for n in FN:
        row = df[df["filename"].apply(_norm) == _norm(n)]
        if row.empty:
            print(f"  {n}: not in labels"); continue
        r = row.iloc[0]
        ts = r["ts_sec"]
        if ts is None or pd.isna(ts):
            print(f"  {n}: no timestamp, skip"); continue
        ts = int(ts)
        f = find_file(n)
        if f is None:
            print(f"  {n}: file not found"); continue
        rgb = grab_frame_at(f, ts)
        if rgb is None:
            print(f"  {n}: frame grab failed at t={ts}"); continue
        img = Image.fromarray(rgb)
        img_path = OUT_DIR / f"{n}_t{ts}.jpg"
        img.save(img_path, quality=92)
        paths_for_montage.append(str(img_path))
        labels_for_montage.append(f"{n}  t={ts}s")

        enc = vlm.encode_image(img)
        q1 = vlm.answer_question(enc, "Describe what the person is doing in this image in one sentence.", tok)
        q2 = vlm.answer_question(enc, "How many people are in the image? Where are they?", tok)
        q3 = vlm.answer_question(enc, "Are any hands visible? What are they touching or holding?", tok)
        q4 = vlm.answer_question(enc, "Is there any cash, banknotes, or bundles of money visible?", tok)

        entry = {
            "video": n, "t_sec": ts, "file": f.name, "frame": str(img_path),
            "describe": q1.strip(), "people": q2.strip(),
            "hands": q3.strip(), "cash": q4.strip(),
        }
        results.append(entry)
        print(f"\n=== {n}  t={ts}s ===")
        print(f"  describe: {entry['describe']}")
        print(f"  people:   {entry['people']}")
        print(f"  hands:    {entry['hands']}")
        print(f"  cash:     {entry['cash']}", flush=True)

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if paths_for_montage:
        mp = make_montage(paths_for_montage, labels_for_montage, OUT_DIR / "montage_all.jpg", cols=4)
        print(f"\nMontage: {mp}")
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
