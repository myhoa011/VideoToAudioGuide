from pydantic import BaseModel
from typing import Optional

class AudioResponse(BaseModel):
    audio_path: str
    duration: Optional[float] = None
    text: str 