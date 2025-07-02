from abc import ABC, abstractmethod
from typing import Union

from PIL import Image
from pydantic import BaseModel
from torch import Tensor


class ObjectDetectionResult(BaseModel):
    bboxes: Tensor
    confidences: Tensor

    class Config:
        arbitrary_types_allowed = True

class ObjectDetectionAdapter(ABC):
    @abstractmethod
    def predict(
        self,
        image: Union[str, Image.Image],
        box_threshold: float,
        iou_threshold: float,
        output_format: str
    ) -> ObjectDetectionResult:
        pass