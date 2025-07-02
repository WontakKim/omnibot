from typing import List, Union

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import ToPILImage

from omniparser.data.som import SOM
from omniparser.util.box_annotator import BoxAnnotator


def get_cropped_image(image: Union[str, Image.Image], bboxes: Tensor, image_size=None):
    if isinstance(image, str):
        image = Image.open(image)

    to_pil = ToPILImage()
    image_np = np.asarray(image)
    h, w = image_np.shape[:2]

    whwh = torch.tensor([w, h, w, h], dtype=torch.float32, device=bboxes.device)
    coords = (bboxes * whwh).int()
    coords = coords.cpu()

    cropped_images = []

    for bbox in coords:
        x_min, y_min, x_max, y_max = bbox.tolist()
        cropped_image = image_np[y_min:y_max, x_min:x_max]
        if image_size is not None:
            cropped_image = cv2.resize(cropped_image, image_size)
        cropped_images.append(to_pil(cropped_image))
    return cropped_images

def create_annotated_image(image: Union[str, Image.Image], labeled_elements: List[SOM]):
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert('RGB')
    image = np.asarray(image)
    h, w, _ = image.shape

    overlay_ratio = h / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * overlay_ratio,
        'text_thickness': max(int(2 * overlay_ratio), 1),
        'text_padding': max(int(3 * overlay_ratio), 1),
        'thickness': max(int(3 * overlay_ratio), 1),
    }

    bboxes = [element.bbox for element in labeled_elements]
    bboxes = bboxes * np.array([w, h, w, h])
    detections = sv.Detections(bboxes)

    labels = [str(i) for i, _ in enumerate(labeled_elements)]

    annotator = BoxAnnotator(**draw_bbox_config)
    annotated_frame = image.copy()
    annotated_frame = annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
        image_size=(w, h)
    )
    pil_image = Image.fromarray(annotated_frame)
    return pil_image

