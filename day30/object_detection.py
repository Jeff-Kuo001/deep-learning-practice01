"""物体检测：交并比与非极大值抑制。"""
import torch

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    upper_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    lower_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inter = (lower_right - upper_left).clamp_min(0).prod(dim=2)
    return inter / (area1[:, None] + area2 - inter).clamp_min(1e-12)

def nms(boxes, scores, threshold=0.5):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][iou <= threshold]
    return torch.tensor(keep)

boxes = torch.tensor([[0., 0., 2., 2.], [0.2, 0.2, 2.1, 2.1], [3., 3., 4., 4.]])
scores = torch.tensor([0.9, 0.8, 0.7])
print("IoU:\n", box_iou(boxes, boxes))
print("NMS keeps:", nms(boxes, scores).tolist())
