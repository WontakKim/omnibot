from dataclasses import dataclass

import numpy as np


# SOM: set of mark
@dataclass
class SOM:
    type: str
    bbox: np.ndarray
    interactivity: bool
    content: str | None
    source: str
