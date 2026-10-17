"""Diagnostic: for each FN/FP video, dump YOLO people count + VLM answers
   at and around the labeled timestamp. Robust to cv2 5.x NaN FPS."""
import cv2, re, sys, json, subprocess
from pathlib import Path
import pandas as pd
from engine import CashSecurityEngine

VIDEO_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
LABELS    = Path("/Users/almansurkakimov/Downloads/open/open_labels.xlsx")
OUT_JSON  = Path("baseline/fn_fp_census.json")

FN = ["Х-11","Х-13","Х-15","Х-20","Х-22","Х-24","Х-26","Х-28","Х-32","Х-40","Х-44","Х-46"]
FP = ["Х-3","Х-5","Х-16","Х-19","Х-35","Х-37","Х-39","Х-48"]

def parse_ts(s):
    if pd.isna(s): return None
    s = str(s).lower()
    m = re.search(r"(\d+)\s*минут\w*\s*(\d+)", s)
    if m: return int(m.group(1))*60 + int(m.group(2))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def _norm(x):
    return str(x).replace("Х","X").replace("х","X").strip()

def find_file(name):
    n = _norm(name)
    for f in VIDEO_DIR.glob("*.mov"):
        if _norm(f.stem) == n:
            return f
    return None

def ffprobe_duration(video):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0", str(video)],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip())
    except Exception:
        return None

def probe(engine, video, times_sec, ffprobe_dur):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps != fps or fps <= 0:  # NaN or invalid
        fps = 30.0
    out = {}
    for t in times_sec:
        if t is None or (isinstance(t, float) and t != t):
            continue
        if ffprobe_dur is not None and t > ffprobe_dur:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
        ok, frame = cap.read()
        if not ok:
            out[int(t)] = {"error": "read_failed"}
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = engine.yolo.predict(frame_rgb, classes=[0], device=engine.device, verbose=False)
        n_people = len(res[0].boxes)
        for k in engine.violation_streaks:
            engine.violation_streaks[k] = {"count":0,"first_seen_sec":0,"reported":False}
        _, viols = engine.analyze_frame(frame_rgb, int(t))
        out[int(t)] = {
            "people": n_people,
            "fired_rules": [v["type"] for v in viols] if viols else [],
        }
    cap.release()
    return out

def main():
    df = pd.read_excel(LABELS)
    df.columns = ["filename","ts_raw","label"]
    df["ts_sec"] = df["ts_raw"].apply(parse_ts)

    report = {"FN": {}, "FP": {}}
    if OUT_JSON.exists():
        try:
            report = json.loads(OUT_JSON.read_text())
            report.setdefault("FN", {}); report.setdefault("FP", {})
        except Exception as e:
            print(f"[warn] could not read prior {OUT_JSON}: {e}")

    print("Loading engine (YOLOv8n + Moondream2) ...", flush=True)
    eng = CashSecurityEngine()

    for bucket, names in [("FN", FN), ("FP", FP)]:
        for n in names:
            if n in report[bucket] and "probes" in report[bucket][n]:
                print(f"[{bucket}] {n} cached, skip", flush=True); continue
            try:
                row = df[df["filename"].apply(_norm) == _norm(n)]
                if row.empty:
                    report[bucket][n] = {"error":"not in labels"}
                    print(f"[{bucket}] {n} NOT IN LABELS", flush=True); continue
                r = row.iloc[0]
                ts = r["ts_sec"]
                f = find_file(n)
                if f is None:
                    report[bucket][n] = {"error":"file not found"}
                    print(f"[{bucket}] {n} FILE NOT FOUND", flush=True); continue
                dur = ffprobe_duration(f)
                if ts is None:
                    if dur:
                        probe_times = sorted(set(int(x) for x in
                            [dur*0.15, dur*0.35, dur*0.55, dur*0.75, dur*0.90]))
                    else:
                        probe_times = [5, 10, 15, 20, 30]
                else:
                    probe_times = sorted(set([max(1, ts-3), ts, ts+3]))

                report[bucket][n] = {
                    "true_label": r["label"],
                    "true_ts": ts,
                    "file": f.name,
                    "ffprobe_dur": dur,
                    "probes": probe(eng, f, probe_times, dur),
                }
                print(f"[{bucket}] {n} done  (dur={dur}, probes={probe_times})", flush=True)
            except Exception as e:
                report[bucket][n] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[{bucket}] {n} ERROR {type(e).__name__}: {e}", flush=True)
            finally:
                OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n===== COMPACT SUMMARY =====")
    for bucket in ("FN","FP"):
        print(f"\n--- {bucket} ---")
        for name, info in report[bucket].items():
            if "error" in info:
                print(f"  {name}: ERROR {info['error']}")
                continue
            print(f"  {name}  true_ts={info['true_ts']}  dur={info.get('ffprobe_dur')}  file={info['file']}")
            for t, p in info["probes"].items():
                if "error" in p:
                    print(f"     t={t:>3}  ERROR {p['error']}")
                else:
                    print(f"     t={int(t):>3}s  people={p['people']}  fired={p['fired_rules'] or '-'}")
    print(f"\nWrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
