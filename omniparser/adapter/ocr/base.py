from abc import ABC, abstractmethod
from typing import Union, List

from PIL.Image import Image
from pydantic import BaseModel
from torch import Tensor


class OCRResult(BaseModel):
    texts: List[str]
    bboxes: Tensor

    class Config:
        arbitrary_types_allowed = True

class OCRAdapter(ABC):
    @abstractmethod
    def extract_text(
        self,
        image: Union[str, Image],
        output_format: str
    ) -> OCRResult:
        pass