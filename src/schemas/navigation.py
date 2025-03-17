from pydantic import BaseModel
from typing import List
from src.schemas.depth import ObjectWithDepth

class NavigationGuide(BaseModel):
    navigation_text: str
    priority_objects: List[ObjectWithDepth] 