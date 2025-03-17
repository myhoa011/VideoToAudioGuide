from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class VideoFrame(BaseModel):
    timestamp: str
    video_path: str

class VideoProcessingResult(BaseModel):
    status: str
    video_path: str 
    total_frames: int
    frames: List[VideoFrame]

class VideoFolder(BaseModel):
    folder_name: str
    frame_count: int
    created_at: Optional[datetime] = None