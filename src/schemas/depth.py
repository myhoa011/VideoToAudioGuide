from pydantic import BaseModel
from src.schemas.detection import DetectedObject

class ObjectWithDepth(DetectedObject):
    depth: float  # 0-1 scale
    distance_rank: int