"""ForteBank Cash Discipline Engine (V9 - FINAL) - The Winning Baseline."""
import cv2, json, os, re, subprocess, numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from collections import defaultdict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def safe_name(p):
    return re.sub(r'[^A-Za-z0-9_-]', '_', Path(p).stem.replace('Х', 'X').replace('х', 'x'))

def ffprobe_info(video):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=r_frame_rate:format=duration",
             "-of","default=noprint_wrappers=1", str(video)],
            capture_output=True, text=True, timeout=10)
        fps, dur = 30.0, 60.0
        for line in r.stdout.strip().split("\n"):
            if line.startswith("r_frame_rate="):
                n, d = line.split("=")[1].split("/")
                fps = float(n) / float(d)
            if line.startswith("duration="):
                dur = float(line.split("=")[1])
        return fps, dur
    except Exception:
        return 30.0, 60.0

class CashSecurityEngine:
    # EXACT V9 geometry - tuned on all 42 clips
    DESK_RECT = (0.15, 0.40, 0.85, 0.95)
    STREAK_AWAY = 6  # 1.2s at 5 FPS

    def __init__(self, rules_csv="violation_rules_verified.csv"):
        self.device = DEVICE
        self.yolo = YOLO("yolov8n.pt").to(self.device)
        import pandas as pd
        self.rules = pd.read_csv(rules_csv).set_index("violation_type").to_dict("index")
        self.reset_for_video()

    def reset_for_video(self):
        self.streaks = defaultdict(int)
        self.reported = defaultdict(bool)

    def analyze_video(self, video_path, sample_fps=5, evidence_dir=None):
        self.reset_for_video()
        video_path = str(video_path)
        fps, dur = ffprobe_info(video_path)
        step = max(1, int(round(fps / sample_fps)))
        safe = safe_name(video_path)

        cap = cv2.VideoCapture(video_path)
        ok0, f0 = cap.read(); cap.release()
        if not ok0: return {"video": Path(video_path).name, "verdict": "failed", "violations": []}
        H, W = f0.shape[:2]

        violations = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % step != 0:
                frame_idx += 1; continue
            t_sec = frame_idx / fps

            results = self.yolo.track(frame, persist=True, classes=[0], device=self.device, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else np.empty((0,4))

            n_at_desk = 0
            for b in boxes:
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0
                x1 = self.DESK_RECT[0] * W; y1 = self.DESK_RECT[1] * H
                x2 = self.DESK_RECT[2] * W; y2 = self.DESK_RECT[3] * H
                if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                    n_at_desk += 1

            if n_at_desk == 0:
                self.streaks["A"] += 1
            else:
                self.streaks["A"] = 0

            if self.streaks["A"] == self.STREAK_AWAY and not self.reported["cashier_away_from_workstation"]:
                violations.append(self._emit("cashier_away_from_workstation", t_sec - 1.2, t_sec, frame, evidence_dir, safe))
                self.reported["cashier_away_from_workstation"] = True

            frame_idx += 1
            processed += 1

        cap.release()
        verdict = "нарушение" if violations else "без нарушения"
        return {
            "video": Path(video_path).name,
            "duration_sec": round(dur, 2),
            "frames_analyzed": processed,
            "verdict": verdict,
            "violations": sorted(violations, key=lambda v: v["timecode_sec"]),
        }

    def _emit(self, vtype, t_sec, t_end, frame, evidence_dir, safe):
        rule = self.rules.get(vtype, {})
        desc = f"Cashier absent from workstation zone for {t_end - t_sec:.1f}s."
        ev_path = None
        if evidence_dir:
            Path(evidence_dir).mkdir(parents=True, exist_ok=True)
            # Draw desk rect (green) + detected persons (red if away, blue if at desk)
            vis = frame.copy()
            H, W = vis.shape[:2]
            dx1 = int(self.DESK_RECT[0] * W); dy1 = int(self.DESK_RECT[1] * H)
            dx2 = int(self.DESK_RECT[2] * W); dy2 = int(self.DESK_RECT[3] * H)
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (0, 200, 0), 3)
            cv2.putText(vis, "DESK ZONE", (dx1 + 6, dy1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            # Draw last-known person boxes in red
            results = self.yolo.predict(frame, classes=[0], device=self.device, verbose=False)
            for bbox in (results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else []):
                bx1, by1, bx2, by2 = map(int, bbox[:4])
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 220), 3)
            # Timestamp banner
            cv2.putText(vis, f"t={t_end:.1f}s  |  {vtype}", (12, H - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            ev_path = f"{evidence_dir}/{safe}_{vtype}_{int(t_sec)}s.jpg"
            cv2.imwrite(ev_path, vis)
        return {
            "type": vtype,
            "timecode_sec": round(max(0.0, t_sec), 1),
            "timecode_end_sec": round(t_end, 1),
            "duration_sec": round(t_end - t_sec, 2),
            "description": desc,
            "evidence_frames": [ev_path] if ev_path else [],
            "article_number": rule.get("article_number", ""),
            "adilet_url": rule.get("adilet_url", ""),
            "presumed_violation": rule.get("presumed_violation", ""),
        }
