from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List

import numpy as np
from PIL.Image import Image


@dataclass
class OCRResult:
    texts: List[str]
    bboxes: np.ndarray

class OCRAdapter(ABC):
    @abstractmethod
    def extract_text(
        self,
        image: Union[str, Image],
        output_format: str='xyxy'
    ) -> OCRResult:
        pass