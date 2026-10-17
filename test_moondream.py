import time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_moondream():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Moondream2 (1.8B) natively on {device.upper()}...")
    
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26" # Locking the revision for stability
    
    # Load model and tokenizer directly into the M1 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        revision=revision
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    print("Model loaded! Reading image...")
    image = Image.open("zone_debug.jpg")
    
    # Moondream performs best with direct, specific questions
    questions = [
        "Is the second person standing in the background inside the room, or behind a glass door?",
        "Look at the grey safe box on the right. Is its door open or closed?"
    ]
    
    print("\nEncoding image and running inference...")
    start_time = time.time()
    
    enc_image = model.encode_image(image)
    
    for q in questions:
        answer = model.answer_question(enc_image, q, tokenizer)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    
    duration = time.time() - start_time
    print(f"\n✅ Total VLM Response Time: {duration:.2f} seconds")

if __name__ == "__main__":
    test_moondream()