from pydantic import BaseModel
from typing import Optional

class AudioResponse(BaseModel):
    audio_data: Optional[bytes] = None
    text: str
    duration: Optional[float] = None
    voice: Optional[str] = None
    format: Optional[str] = 'wav'
    engine: Optional[str] = 'openai' 