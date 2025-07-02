from typing import List, Union

from pydantic import BaseModel
from torch import Tensor


# SOM: set of mark
class SOM(BaseModel):
    type: str
    bbox: Union[List[float], Tensor]
    interactivity: bool
    content: str | None
    source: str

    def safe(self):
        if isinstance(self.bbox, Tensor):
            self.bbox = self.bbox.detach().cpu().tolist()
        return self

    class Config:
        arbitrary_types_allowed = True
