"""边界框实现：角点格式与中心格式相互转换。"""
import torch

def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1), dim=-1)

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2), dim=-1)

boxes = torch.tensor([[10., 20., 50., 80.], [5., 5., 25., 35.]])
center = box_corner_to_center(boxes)
print("center format:\n", center)
print("restored:\n", box_center_to_corner(center))
