"""锚框：在每个像素中心生成不同尺度和宽高比的先验框。"""
import torch

def multibox_prior(feature_map, sizes, ratios):
    height, width = feature_map.shape[-2:]
    device = feature_map.device
    boxes_per_pixel = len(sizes) + len(ratios) - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[:1]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * height / width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[:1]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    offsets = torch.stack((-w, -h, w, h), dim=1).repeat(height * width, 1) / 2
    cy = (torch.arange(height, device=device) + 0.5) / height
    cx = (torch.arange(width, device=device) + 0.5) / width
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")
    centers = torch.stack((gx, gy, gx, gy), dim=-1).reshape(-1, 4)
    anchors = centers.repeat_interleave(boxes_per_pixel, 0) + offsets.repeat(height * width, 1)
    return anchors.unsqueeze(0)

anchors = multibox_prior(torch.zeros(1, 3, 4, 4), [0.3, 0.5], [1, 2, 0.5])
print("anchor shape:", tuple(anchors.shape))
print(anchors[0, :3])
