from abc import ABC, abstractmethod

from PIL import Image


class VisionLanguageAdapter(ABC):
    def hello(self):
        print("Hello")
    # @abstractmethod
    # def process(self, image: Image.Image, **kwargs):
    #     pass
