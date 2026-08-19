"""硬件：CPU 和 GPU 的设备信息与张量计算。"""
import time
import torch

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))

for device in devices:
    x = torch.randn(1024, 1024, device=device)
    start = time.perf_counter()
    _ = x @ x
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(device, "elapsed:", round(elapsed, 4), "seconds")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))
