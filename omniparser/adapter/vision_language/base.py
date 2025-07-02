from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class VisionLanguageAdapter(ABC):
    @abstractmethod
    def gen_text(
        self,
        images: List[Image.Image],
        prompt: str
    ) -> List[str]:
        pass
