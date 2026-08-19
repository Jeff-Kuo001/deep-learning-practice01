"""分布式训练：DistributedDataParallel 的最小结构。"""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = DDP(nn.Linear(8, 2).to(device), device_ids=[local_rank] if device.type == "cuda" else None)
    loss = model(torch.randn(4, 8, device=device)).sum()
    loss.backward()
    print("rank", rank, "finished")
    dist.destroy_process_group()

if __name__ == "__main__":
    if "RANK" in os.environ:
        main()
    else:
        print("run with: torchrun --standalone --nproc_per_node=2 distributed_training.py")
