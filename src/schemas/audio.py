from pydantic import BaseModel
from typing import Optional

class AudioResponse(BaseModel):
    audio_path: str
    text: str
    duration: Optional[float] = None
    voice: Optional[str] = None
    format: Optional[str] = 'wav'
    engine: Optional[str] = 'openai' 