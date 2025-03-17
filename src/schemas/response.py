from pydantic import BaseModel
from typing import List
from src.schemas.depth import ObjectWithDepth
from src.schemas.navigation import NavigationGuide
from src.schemas.audio import AudioResponse

class ExecutionTime(BaseModel):
    object_detection: float = 0
    depth_estimation: float = 0
    navigation_generation: float = 0
    text_to_speech: float = 0
    total: float = 0

class FrameAnalysis(BaseModel):
    frame_index: str
    frame_path: str
    objects: List[ObjectWithDepth]
    navigation: NavigationGuide
    audio: AudioResponse
    execution_time: ExecutionTime

class VideoAnalysisResponse(BaseModel):
    video_path: str
    total_frames: int
    frames_analysis: List[FrameAnalysis] 