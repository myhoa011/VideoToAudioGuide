from pydantic import BaseModel, Field
from typing import List

class DetectedObject(BaseModel):
    box_2d: List[float]  # [y_min, x_min, y_max, x_max]
    label: str  # "person", "car", etc.
    position: str  # "left", "right", "center"
    type: str = Field(description="")  # "vehicle", "person", "animal", etc. 
