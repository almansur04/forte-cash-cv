import cv2
import torch
import time
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer

class CashSecurityEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.yolo = YOLO("yolov8n.pt")
        
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        self.vlm = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        
        self.violation_streaks = {
            "unauthorized_person": {"count": 0, "first_seen_sec": 0, "reported": False},
            "unattended_open_safe": {"count": 0, "first_seen_sec": 0, "reported": False},
            "unattended_workstation": {"count": 0, "first_seen_sec": 0, "reported": False} # NEW RULE
        }
        self.STREAK_THRESHOLD = 3 

    def analyze_frame(self, frame_rgb, current_sec):
        results = self.yolo.predict(frame_rgb, classes=[0], device=self.device, verbose=False)
        people_count = len(results[0].boxes)

        pil_img = Image.fromarray(frame_rgb)
        
        # --- Rule 1: Unauthorized Person (2+ people) ---
        if people_count >= 2:
            enc_img = self.vlm.encode_image(pil_img)
            prompt = "Are there two people standing inside the same room? (Ignore anyone standing behind a glass wall). Answer strictly yes or no."
            ans = self.vlm.answer_question(enc_img, prompt, self.tokenizer).strip().lower()
            if "yes" in ans:
                if self.violation_streaks["unauthorized_person"]["count"] == 0:
                    self.violation_streaks["unauthorized_person"]["first_seen_sec"] = current_sec
                self.violation_streaks["unauthorized_person"]["count"] += 1
            else:
                self.violation_streaks["unauthorized_person"]["count"] = 0
                self.violation_streaks["unauthorized_person"]["reported"] = False
        else:
            self.violation_streaks["unauthorized_person"]["count"] = 0
            self.violation_streaks["unauthorized_person"]["reported"] = False

        # --- Rule 2: Unattended Open Safe (0 people) ---
        if people_count == 0:
            enc_img = self.vlm.encode_image(pil_img)
            ans = self.vlm.answer_question(enc_img, "Is the grey safe box door visibly open? Answer strictly yes or no.", self.tokenizer).strip().lower()
            if "yes" in ans:
                if self.violation_streaks["unattended_open_safe"]["count"] == 0:
                    self.violation_streaks["unattended_open_safe"]["first_seen_sec"] = current_sec
                self.violation_streaks["unattended_open_safe"]["count"] += 1
            else:
                self.violation_streaks["unattended_open_safe"]["count"] = 0
                self.violation_streaks["unattended_open_safe"]["reported"] = False
        else:
            self.violation_streaks["unattended_open_safe"]["count"] = 0
            self.violation_streaks["unattended_open_safe"]["reported"] = False

        # --- Rule 3: Unattended Active Workstation (1 person, walking away) ---
        if people_count == 1:
            enc_img = self.vlm.encode_image(pil_img)
            # First, check if they left the chair
            ans_sitting = self.vlm.answer_question(enc_img, "Is the person sitting down at the wooden desk? Answer strictly yes or no.", self.tokenizer).strip().lower()
            
            if "no" in ans_sitting:
                # If they are not at the desk, check if the PC is left on
                ans_pc = self.vlm.answer_question(enc_img, "Is the laptop computer screen open and brightly lit? Answer strictly yes or no.", self.tokenizer).strip().lower()
                if "yes" in ans_pc:
                    if self.violation_streaks["unattended_workstation"]["count"] == 0:
                        self.violation_streaks["unattended_workstation"]["first_seen_sec"] = current_sec
                    self.violation_streaks["unattended_workstation"]["count"] += 1
                else:
                    self.violation_streaks["unattended_workstation"]["count"] = 0
            else:
                self.violation_streaks["unattended_workstation"]["count"] = 0
                self.violation_streaks["unattended_workstation"]["reported"] = False
        else:
            self.violation_streaks["unattended_workstation"]["count"] = 0
            self.violation_streaks["unattended_workstation"]["reported"] = False

        # --- Evaluate Streaks ---
        final_violations = []
        for v_type, state in self.violation_streaks.items():
            if state["count"] >= self.STREAK_THRESHOLD and not state["reported"]:
                final_violations.append({
                    "type": v_type,
                    "evidence": f"Detected continuously for {self.STREAK_THRESHOLD} seconds",
                    "timecode_sec": state["first_seen_sec"]
                })
                state["reported"] = True 

        return people_count, final_violations