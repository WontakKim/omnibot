from typing import Union

from PIL import Image
from ultralytics import YOLO

from omniparser.adapter.object_detection.base import ObjectDetectionAdapter, ObjectDetectionResult


class YoloAdapter(ObjectDetectionAdapter):
    def __init__(
        self,
        model_path: str
    ):
        self.model = YOLO(model_path)

    def predict(
        self,
        image: Union[str, Image.Image],
        box_threshold: float=0.1,
        iou_threshold: float=0.7,
        output_format: str='xyxy'
    ):
        result = self.model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold
        )[0]
        return ObjectDetectionResult(
            bboxes=(result.boxes.xyxy if output_format == 'xyxy' else result.boxes.xywh).cpu().numpy(),
            confidences=result[0].boxes.conf.cpu().numpy(),
        )
