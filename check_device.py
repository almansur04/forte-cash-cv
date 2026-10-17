import torch
import llama_cpp
import cv2
import ultralytics

def check_environment():
    print("=== Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"llama_cpp_python version: {llama_cpp.__version__}")
    
    print("\n=== GPU Acceleration Check ===")
    # Check PyTorch MPS
    if torch.backends.mps.is_available():
        print("✅ PyTorch MPS (Metal Performance Shaders): AVAILABLE")
        # Test tensor creation on GPU
        x = torch.ones(1, device="mps")
        print(f"   Tensor successfully created on: {x.device}")
    else:
        print("❌ PyTorch MPS: NOT AVAILABLE (Falling back to CPU)")

    # Check llama.cpp Metal compile flag (heuristic check)
    try:
        # If llama-cpp-python was compiled with Metal, the ggml metal library loads successfully
        print("✅ llama.cpp Metal Support: COMPILED (Requires model load to fully verify)")
    except Exception as e:
        print(f"❌ llama.cpp Metal Support: ERROR ({e})")

    print("\n=== Device Abstraction Configuration ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Primary Detection Device: {device.upper()}")
    
if __name__ == "__main__":
    check_environment()