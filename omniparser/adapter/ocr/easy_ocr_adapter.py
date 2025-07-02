from typing import List, Union

import easyocr
import numpy as np
import torch
from PIL import Image

from omniparser.adapter.ocr.base import OCRAdapter, OCRResult


class EasyOCRAdapter(OCRAdapter):
    def __init__(
        self,
        lang_list: List[str],
        device: str
    ):
        self.device = device
        self.reader = easyocr.Reader(lang_list)

    def extract_text(
        self,
        image: Union[str, Image.Image],
        output_format: str='xyxy'
    ) -> OCRResult:
        if isinstance(image, str):
            image = Image.open(image)

        if image.mode == 'RGBA':
            # Convert RGBA to RGB to avoid alpha channel issues
            image = image.convert('RGB')

        image_np = np.asarray(image)

        # result format
        # [([[left, top], [right, top], [left, bottom], [right, bottom]], 'ocr text', confidence), ...]
        result = self.reader.readtext(image_np)
        format_func = get_xyxy if output_format == 'xyxy' else get_xywh
        bboxes, texts = zip(*[(format_func(item[0]), item[1]) for item in result]) if result else ([], [])
        return OCRResult(
            texts=texts,
            bboxes=torch.tensor(bboxes, dtype=torch.float32, device=self.device)
        )

def get_xyxy(value):
    x, y, xp, yp = value[0][0], value[0][1], value[2][0], value[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh(value):
    x, y, w, h = value[0][0], value[0][1], value[2][0] - value[0][0], value[2][1] - value[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h
