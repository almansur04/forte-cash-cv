import cv2
import torch
import time
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer

VIDEO_PATH = "/Users/almansurkakimov/Downloads/open/videos/X-1.mov"

class CashSecurityEngine:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing Hybrid Engine on {self.device.upper()}...")

        # 1. Load YOLO (Fast Tracking)
        self.yolo = YOLO("yolov8n.pt")

        # 2. Load Moondream2 (Deep VQA)
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        self.vlm = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        
        print("Models loaded successfully into RAM.")

    def analyze_frame(self, frame_rgb):
        # 1. Fast Pass: Count people with YOLO
        results = self.yolo.predict(frame_rgb, classes=[0], device=self.device, verbose=False)
        people_count = len(results[0].boxes)

        violations = []
        pil_img = Image.fromarray(frame_rgb)
        
        # 2. Smart Triggers for Heavy Pass (VLM)
        if people_count >= 2:
            enc_img = self.vlm.encode_image(pil_img)
            answer = self.vlm.answer_question(enc_img, "Are there two people standing inside the same room? Answer strictly yes or no.", self.tokenizer)
            if "yes" in answer.lower():
                violations.append({"type": "unauthorized_person", "evidence": answer})

        elif people_count == 0:
            enc_img = self.vlm.encode_image(pil_img)
            ans_safe = self.vlm.answer_question(enc_img, "Is the grey safe box open? Answer strictly yes or no.", self.tokenizer)
            if "yes" in ans_safe.lower():
                violations.append({"type": "unattended_open_safe", "evidence": ans_safe})

        return people_count, violations

    def test_run(self, video_path, max_seconds=5):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) # Extract 1 frame per second
        
        frame_idx = 0
        seconds_processed = 0
        start_time = time.time()
        
        print(f"\nStarting video analysis at 1 FPS (Max {max_seconds} seconds)...")
        while cap.isOpened() and seconds_processed < max_seconds:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # OpenCV uses BGR, models expect RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(f"\n--- Second {seconds_processed + 1} ---")
                
                p_count, frame_violations = self.analyze_frame(frame_rgb)
                print(f"YOLO People Count: {p_count}")
                
                if frame_violations:
                    print(f"⚠️ VIOLATION DETECTED: {frame_violations}")
                else:
                    print("✅ Rule Check Passed")
                
                seconds_processed += 1
            frame_idx += 1
            
        cap.release()
        print(f"\nProcessing finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    engine = CashSecurityEngine()
    engine.test_run(VIDEO_PATH, max_seconds=5)