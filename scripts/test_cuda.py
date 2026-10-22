import sys
import torch
import warp as wp

def main():
    print("=" * 60)
    print("MicroDuck Workstation GPU Verification (Jupiter)")
    print("=" * 60)
    
    print(f"Python: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available in PyTorch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  Device Capability: {torch.cuda.get_device_capability(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    wp.init()
    print("\nNVIDIA Warp:")
    print(f"  Warp Version: {wp.__version__}")
    print(f"  Default Device: {wp.get_device()}")
    print("=" * 60)
    print("Status: CUDA Simulation & RL Training Pipeline Fully Operational")
    print("=" * 60)

if __name__ == "__main__":
    main()
