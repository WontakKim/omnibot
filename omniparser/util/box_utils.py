import numpy as np


def box_inclusion(box1, box2, threshold=0.8, eps=1e-7):
    inner_area = box_area(box1) # [N]
    inter, _ = box_inter_union(box1, box2) # [N,_]
    ratio = inter / (inner_area[:, None] + eps)
    return ratio > threshold

def box_area(box: np.ndarray):
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

def box_inter_union(box1: np.ndarray, box2: np.ndarray):
    area1 = box_area(box1) # [N]
    area2 = box_area(box2) # [M]

    lt = np.maximum(box1[:, None, :2], box2[:, :2]) # [N,M,2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:]) # [N,M,2]
    wh = np.clip(rb - lt, 0, None) # [N,M,2]

    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter, union

def box_iou_custom(box1: np.ndarray, box2: np.ndarray, eps=1e-7):
    area1 = box_area(box1)
    area2 = box_area(box2)
    inter, union = box_inter_union(box1, box2)
    iou_matrix = inter / (union + eps)
    # check inclusion
    ratio1 = inter / (area1 + eps)
    ratio2 = inter / (area2 + eps)
    return np.maximum(iou_matrix, np.maximum(ratio1, ratio2))

def remove_overlap(box: np.ndarray, iou_threshold=0.9):
    iou_matrix = box_iou_custom(box, box)
    np.fill_diagonal(iou_matrix, 0)
    area = box_area(box)
    area_compare = area[:, None] > area[None, :]
    overlap = np.any((iou_matrix > iou_threshold) & area_compare, axis=1)
    keep = ~overlap
    return keep

# no implemented
def batch_remove_overlap():
    pass
