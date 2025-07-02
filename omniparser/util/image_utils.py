from typing import List, Union

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from torchvision.transforms.v2 import ToPILImage

from omniparser.data.som import SOM
from omniparser.util.box_annotator import BoxAnnotator


def get_cropped_image(image: Union[str, Image.Image], bboxes, image_size=None):
    if isinstance(image, str):
        image = Image.open(image)

    to_pil = ToPILImage()
    image = np.asarray(image)
    h, w = image.shape[:2]

    coords = (bboxes * np.array([w, h, w, h])).astype(int)
    cropped_images = []

    for x_min, y_min, x_max, y_max in coords:
        cropped_image = image[y_min:y_max, x_min:x_max]
        if image_size is not None:
            cropped_image = cv2.resize(cropped_image, image_size)
        cropped_images.append(to_pil(cropped_image))
    return cropped_images

def create_annotated_image(image: Union[str, Image.Image], labeled_elements: List[SOM]):
    if isinstance(image, str):
        image = Image.open(image)

    def annotated_text(i, text):
        text = text if len(text) <= 10 else text[:7] + '...'
        return f'{i}: {text}'

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

    labels = [annotated_text(i, element.content) for i, element in enumerate(labeled_elements)]

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

