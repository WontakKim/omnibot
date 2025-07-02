from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image


@dataclass
class ObjectDetectionResult:
    bboxes: np.ndarray
    confidences: np.ndarray

class ObjectDetectionAdapter(ABC):
    @abstractmethod
    def predict(
        self,
        image: Union[str, Image.Image],
        box_threshold: float,
        iou_threshold: float,
        output_format: str='xyxy'
    ) -> ObjectDetectionResult:
        pass