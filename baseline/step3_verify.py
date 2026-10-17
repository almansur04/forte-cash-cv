"""Step 3 verification: YOLO+ByteTrack, CLIP zero-shot on real clip."""
import os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch, cv2
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"DEVICE = {DEVICE}")

VIDEO = "/Users/almansurkakimov/Downloads/open/videos/Х-15.mov"
if not Path(VIDEO).exists():
    VIDEO = "/Users/almansurkakimov/Downloads/open/videos/X-15.mov"
assert Path(VIDEO).exists(), f"video not found: {VIDEO}"

print("\n[TEST 1] YOLO + ByteTrack")
t0 = time.time()
yolo = YOLO("yolov8n.pt")
print(f"  YOLO load: {time.time()-t0:.2f}s")

cap = cv2.VideoCapture(VIDEO)
frames_processed = 0
tracks_seen = set()
phone_detections = 0
t_start = time.time()
while cap.isOpened() and frames_processed < 20:
    ok, frame = cap.read()
    if not ok: break
    if frames_processed % 30 == 0:
        results = yolo.track(frame, persist=True, classes=[0, 67], device=DEVICE, verbose=False)
        if results[0].boxes.id is not None:
            for tid, cls in zip(results[0].boxes.id.tolist(), results[0].boxes.cls.tolist()):
                tracks_seen.add((int(tid), int(cls)))
                if int(cls) == 67:
                    phone_detections += 1
    frames_processed += 1
cap.release()
print(f"  analyzed {frames_processed//30 + 1} sampled frames in {time.time()-t_start:.2f}s")
print(f"  unique tracks: {sorted(tracks_seen)}")
print(f"  phone detections: {phone_detections}")
print("  OK" if tracks_seen else "  WARN: no tracks")

print("\n[TEST 2] CLIP zero-shot")
t0 = time.time()
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"  CLIP load: {time.time()-t0:.2f}s")

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ok, frame = cap.read(); cap.release()
assert ok
h, w = frame.shape[:2]
crop = frame[int(h*0.40):int(h*0.85), int(w*0.10):int(w*0.90)]
pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
pil.save("baseline/step3_desk_crop.jpg", quality=90)

labels = [
    "an open cash tray with money", "a closed cash tray",
    "an open safe door", "a closed safe door",
    "an empty wooden desk", "a desk with a computer on it",
]
inputs = proc(text=labels, images=pil, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    logits = clip(**inputs).logits_per_image.softmax(dim=1)[0].tolist()
print("  CLIP scores:")
for lab, sc in sorted(zip(labels, logits), key=lambda x: -x[1]):
    print(f"    {sc:.3f}  {lab}")

print("\n[TEST 3] CLIP timing")
t0 = time.time()
for _ in range(5):
    with torch.no_grad():
        clip(**inputs)
print(f"  5 calls: {time.time()-t0:.2f}s  ({(time.time()-t0)/5*1000:.0f} ms/call)")

print("\n===== STEP 3 VERIFICATION COMPLETE =====")
