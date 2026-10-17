"""ForteBank Cash Discipline Engine (ONNX Optimized)."""
import cv2, json, os, re, subprocess, numpy as np, csv
from pathlib import Path
import onnxruntime as ort
from collections import defaultdict

# ONNX Runtime is CPU-native
DEVICE = "cpu"

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
    DESK_RECT = (0.15, 0.40, 0.85, 0.95)
    STREAK_AWAY = 6  # 1.2s at 5 FPS
    CONF_THRESH = 0.25
    NMS_THRESH = 0.45
    INPUT_SIZE = 416

    def __init__(self, rules_csv="violation_rules_verified.csv"):
        self.device = "cpu"
        self.session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        self.rules = {}
        with open(rules_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                self.rules[row["violation_type"]] = row
        self.reset_for_video()

    def reset_for_video(self):
        self.streaks = defaultdict(int)
        self.reported = defaultdict(bool)

    def _letterbox(self, img):
        shape = img.shape[:2]
        r = min(self.INPUT_SIZE / shape[0], self.INPUT_SIZE / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.INPUT_SIZE - new_unpad[0], self.INPUT_SIZE - new_unpad[1]
        dw /= 2; dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def _infer(self, frame):
        # Preprocess
        img, ratio, (dw, dh) = self._letterbox(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        preds = np.squeeze(outputs[0]).T # (8400, 84)
        
        # Filter class 0 (person)
        scores = preds[:, 4] 
        boxes = preds[:, :4]
        
        mask = scores > self.CONF_THRESH
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            return np.empty((0, 4))
            
        # Convert cx,cy,w,h to x,y,w,h for NMS
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xywh[:, 2] = boxes[:, 2]
        boxes_xywh[:, 3] = boxes[:, 3]
        
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), self.CONF_THRESH, self.NMS_THRESH)
        if len(indices) == 0:
            return np.empty((0, 4))
            
        indices = indices.flatten()
        final_boxes = boxes_xywh[indices]
        
        # Scale back to original frame size
        h, w = frame.shape[:2]
        final_boxes[:, 0] = (final_boxes[:, 0] - dw) / ratio
        final_boxes[:, 1] = (final_boxes[:, 1] - dh) / ratio
        final_boxes[:, 2] = final_boxes[:, 2] / ratio
        final_boxes[:, 3] = final_boxes[:, 3] / ratio
        
        return final_boxes # Returns (N, 4) as x,y,w,h

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

            boxes_xywh = self._infer(frame)
            
            n_at_desk = 0
            for b in boxes_xywh:
                x, y, w, h = b
                cx = x + w / 2
                cy = y + h / 2
                
                x1 = self.DESK_RECT[0] * W; y1 = self.DESK_RECT[1] * H
                x2 = self.DESK_RECT[2] * W; y2 = self.DESK_RECT[3] * H
                
                if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                    n_at_desk += 1

            if n_at_desk == 0:
                self.streaks["A"] += 1
            else:
                self.streaks["A"] = 0

            if self.streaks["A"] == self.STREAK_AWAY and not self.reported["cashier_away_from_workstation"]:
                violations.append(self._emit("cashier_away_from_workstation", t_sec - 1.2, t_sec, frame, boxes_xywh, evidence_dir, safe))
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

    def _emit(self, vtype, t_sec, t_end, frame, boxes_xywh, evidence_dir, safe):
        rule = self.rules.get(vtype, {})
        desc = f"Cashier absent from workstation zone for {t_end - t_sec:.1f}s."
        ev_path = None
        if evidence_dir:
            Path(evidence_dir).mkdir(parents=True, exist_ok=True)
            vis = frame.copy()
            H, W = vis.shape[:2]
            dx1 = int(self.DESK_RECT[0] * W); dy1 = int(self.DESK_RECT[1] * H)
            dx2 = int(self.DESK_RECT[2] * W); dy2 = int(self.DESK_RECT[3] * H)
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (0, 200, 0), 3)
            cv2.putText(vis, "DESK ZONE", (dx1 + 6, dy1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            
            for b in boxes_xywh:
                x, y, w, h = map(int, b)
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 220), 3)
                
            cv2.putText(vis, f"t={t_end:.1f}s  |  {vtype}", (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
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
