import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU detected!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. No GPU detected.")
