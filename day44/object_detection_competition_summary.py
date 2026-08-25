import torch


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    upper_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    lower_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection = (lower_right - upper_left).clamp(min=0).prod(dim=2)
    union = area1[:, None] + area2 - intersection
    return intersection / union


def average_precision(scores, matched):
    order = scores.argsort(descending=True)
    matched = matched[order].float()
    true_positive = matched.cumsum(0)
    false_positive = (1 - matched).cumsum(0)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / max(1, int(matched.sum().item()))
    previous_recall = torch.cat((torch.zeros(1), recall[:-1]))
    return ((recall - previous_recall) * precision).sum()


def mean_average_precision(class_results):
    values = [average_precision(scores, matched) for scores, matched in class_results]
    return torch.stack(values).mean()


if __name__ == "__main__":
    prediction_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.2, 0.2, 0.7, 0.7]])
    target_boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9]])
    print("IoU:", box_iou(prediction_boxes, target_boxes).squeeze())
    results = [
        (torch.tensor([0.9, 0.7, 0.4]), torch.tensor([True, False, True])),
        (torch.tensor([0.8, 0.5]), torch.tensor([True, False])),
    ]
    print("mAP example:", round(mean_average_precision(results).item(), 4))
