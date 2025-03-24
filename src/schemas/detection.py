from pydantic import BaseModel, Field
from typing import List

class DetectedObject(BaseModel):
    box_2d: List[float] = Field(
        description="The 2D bounding box as [y_min, x_min, y_max, x_max]"
    )
    label: str = Field(
        description="General name of the detected object (e.g., 'person', 'car', 'dog')"
    )
    position: str = Field(
        description="Relative position of the object in the image. Must be one of: 'left', 'right', 'center'"
    )
    type: str = Field(
        description="Category or classification of the object (e.g., 'vehicle', 'animal', 'person', 'building')"
    )