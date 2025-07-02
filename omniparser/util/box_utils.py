import torch
import torchvision
from torch import Tensor


def box_inclusion(box1, box2, threshold=0.8, eps=1e-7):
    inner_area = box_area(box1) # [N]
    inter, _ = box_inter_union(box1, box2) # [N,_]
    ratio = inter / (inner_area[:, None] + eps)
    return ratio > threshold

def box_area(box: Tensor):
    return torchvision.ops.box_area(box)

def box_iou_custom(box1: Tensor, box2: Tensor, eps=1e-7):
    area1 = box_area(box1)
    area2 = box_area(box2)
    inter, union = box_inter_union(box1, box2)
    iou_matrix = inter / (union + eps)
    # check inclusion
    ratio1 = inter / (area1 + eps)
    ratio2 = inter / (area2 + eps)
    return torch.maximum(iou_matrix, torch.maximum(ratio1, ratio2))

def remove_overlap(box: Tensor, iou_threshold=0.9):
    iou_matrix = box_iou_custom(box, box)
    iou_matrix.fill_diagonal_(0)
    area = box_area(box)
    area_compare = area[:, None] > area[None, :]
    overlap = torch.any((iou_matrix > iou_threshold) & area_compare, dim=1)
    keep = ~overlap
    return keep

# no implemented
def batch_remove_overlap():
    pass

# torchvision/ops/_utils.py
def upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

# torchvision/ops/boxes.py
def box_inter_union(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union