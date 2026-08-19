"""更多专用硬件：CUDA、MPS 与 XPU 后端检查。"""
import torch

backends = {
    "CUDA": torch.cuda.is_available(),
    "MPS": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    "XPU": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
}
for name, available in backends.items():
    print(f"{name}: {'available' if available else 'unavailable'}")

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.ones(4, dtype=dtype)
    print(dtype, (x + x).dtype)
