import wave
import contextlib
import io

from src.utils.logger import logger
from src.schemas.audio import AudioResponse
from src.utils.constant import (
    OPENAI_MODEL_NAME, OPENAI_TTS_VOICE, 
    TTS_ENGINE_OPENAI, TTS_ENGINE_GTTS, TTS_ENGINE_KOKORO,
    GTTS_LANGUAGE, KOKORO_VOICE, KOKORO_SPEED
)
from src.helpers.tts_helper import (
    call_openai_api, call_gtts, call_kokoro
)
from src.initializer import initializer


class TextToSpeechHandler:
    """Handler for text-to-speech conversion with multiple engine support"""
    
    def __init__(self, engine: str = TTS_ENGINE_KOKORO):
        self.engine = engine
        self.client = initializer.get_openai_client()
        
        # Get TTS engines from initializer
        self.kokoro_pipeline = initializer.get_kokoro_pipeline()
        self.aiogTTS_engine = initializer.get_aiogTTS_engine()
    
    def set_engine(self, engine: str):
        """Change TTS engine"""
        if engine in [TTS_ENGINE_OPENAI, TTS_ENGINE_GTTS, TTS_ENGINE_KOKORO]:
            self.engine = engine
            logger.info(f"TTS engine changed to {engine}")
        else:
            logger.warning(f"Invalid TTS engine {engine}. Using default.")
    
    async def convert_text_to_speech(self, text: str, folder_name: str, frame_index: str, engine: str) -> AudioResponse:
        """
        Convert text to speech using selected engine and return audio data
        
        Args:
            text: Text to convert
            folder_name: Video folder name (for logging)
            frame_index: Frame index (for logging)
            engine: Override default engine (optional)
            
        Returns:
            AudioResponse: Audio response with audio_data
        """
        try:
            # Use provided engine or default
            current_engine = engine if engine else self.engine
            
            logger.info(f"Converting text to speech using {current_engine} for {folder_name}, frame {frame_index}")
            
            voice_used = ""
            audio_data = None
            
            # Call appropriate TTS API based on selected engine
            if current_engine == TTS_ENGINE_OPENAI:
                audio_data = await call_openai_api(self.client, OPENAI_MODEL_NAME, OPENAI_TTS_VOICE, text)
                voice_used = OPENAI_TTS_VOICE
            elif current_engine == TTS_ENGINE_GTTS:
                audio_data = await call_gtts(GTTS_LANGUAGE, text, self.aiogTTS_engine)
                voice_used = f"aiogTTS ({GTTS_LANGUAGE})"
            elif current_engine == TTS_ENGINE_KOKORO:
                audio_data = await call_kokoro(KOKORO_VOICE, KOKORO_SPEED, text, self.kokoro_pipeline)
                voice_used = KOKORO_VOICE
            else:
                logger.warning(f"Unknown engine {current_engine}, falling back to Kokoro")
                audio_data = await call_kokoro(KOKORO_VOICE, KOKORO_SPEED, text, self.kokoro_pipeline)
                voice_used = KOKORO_VOICE
            
            # Get audio duration if possible
            duration = None
            if audio_data:
                try:
                    # Try to get duration from bytes
                    with contextlib.closing(wave.open(io.BytesIO(audio_data), 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)
                except Exception as e:
                    logger.warning(f"Could not get audio duration from bytes: {str(e)}")
            
            audio_response = AudioResponse(
                audio_data=audio_data,
                text=text,
                duration=duration,
                voice=voice_used,
                format='wav',
                engine=current_engine
            )
            
            return audio_response
                
        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            # Return minimal response in case of error
            return AudioResponse(
                audio_data=None,
                text=text,
                engine=engine if engine else self.engine,
                voice="",
                format="wav"
            )