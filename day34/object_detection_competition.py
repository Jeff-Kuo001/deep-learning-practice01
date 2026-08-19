"""目标检测竞赛：预测框的置信度过滤与提交格式。"""
import csv
from io import StringIO
import torch

def format_predictions(image_ids, boxes, scores, labels, threshold=0.5):
    rows = []
    for image_id, image_boxes, image_scores, image_labels in zip(image_ids, boxes, scores, labels):
        keep = image_scores >= threshold
        for box, score, label in zip(image_boxes[keep], image_scores[keep], image_labels[keep]):
            rows.append([image_id, int(label), float(score), *map(float, box)])
    return rows

rows = format_predictions(
    ["image_001"],
    [torch.tensor([[5., 8., 30., 40.], [1., 1., 10., 10.]])],
    [torch.tensor([0.91, 0.22])],
    [torch.tensor([2, 1])],
)
buffer = StringIO()
writer = csv.writer(buffer)
writer.writerow(["image_id", "label", "score", "xmin", "ymin", "xmax", "ymax"])
writer.writerows(rows)
print(buffer.getvalue())
