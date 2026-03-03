"""
GPU 사용 가능 여부 확인
"""
import torch

print("="*60)
print("GPU 정보 확인")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device: {torch.cuda.current_device()}")
else:
    print("CUDA가 사용 불가능합니다.")
    print("GPU 학습을 위해서는 CUDA 지원 PyTorch가 필요합니다.")

print("="*60)
