from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import time
import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

def test_vlm():
    print("Loading Vision Adapter...")
    chat_handler = Llava15ChatHandler(clip_model_path="models/mmproj-model-f16.gguf")
    
    print("Loading LLaVA-1.5-7B on Metal GPU...")
    # n_gpu_layers=-1 forces all layers onto the M1 GPU (Metal)
    llm = Llama(
        model_path="models/ggml-model-q4_k.gguf",
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1, 
        verbose=False
    )
    
    data_uri = image_to_base64_data_uri("zone_debug.jpg")
    
    prompt = """Analyze this image carefully. 
1) Is there a second person visible, and if so, are they in the same room as the cashier, or behind a glass door?
2) Look at the grey safe/box on the right side of the desk. Is its door open or closed?"""

    print("\nSending prompt to VLM (this measures generation speed)...")
    start_time = time.time()
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a precise security monitoring AI."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt}
            ]}
        ],
        max_tokens=150,
        temperature=0.1
    )
    
    duration = time.time() - start_time
    print(f"\n✅ VLM Response Time: {duration:.2f} seconds")
    print("✅ Answer:")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    test_vlm()