from pathlib import Path
import os

import wave
import contextlib

from src.utils.logger import logger
from src.schemas.audio import AudioResponse
from src.utils.constant import (
    OPENAI_MODEL_NAME, OPENAI_TTS_VOICE, OUTPUT_AUDIO_PATH,
    TTS_ENGINE_OPENAI, TTS_ENGINE_GTTS, TTS_ENGINE_KOKORO,
    GTTS_LANGUAGE, KOKORO_VOICE, KOKORO_SPEED
)
from src.helpers.tts_helper import (
    call_openai_api, call_gtts, call_kokoro
)
from src.initializer import initializer


class TextToSpeechHandler:
    """Handler for text-to-speech conversion with multiple engine support"""
    
    def __init__(self, output_path: str = OUTPUT_AUDIO_PATH, engine: str = TTS_ENGINE_KOKORO):
        self.engine = engine
        self.client = initializer.get_openai_client()
        self.base_output_path = Path(output_path)
        self.base_output_path.mkdir(parents=True, exist_ok=True)
        
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
        Convert text to speech using selected engine and save in folder structure
        
        Args:
            text: Text to convert
            folder_name: Video folder name
            frame_index: Frame index
            engine: Override default engine (optional)
            
        Returns:
            AudioResponse: Audio file information
        """
        try:
            # Use provided engine or default
            current_engine = engine if engine else self.engine
            
            # Create a folder for this video's audio files
            video_audio_path = self.base_output_path / folder_name
            video_audio_path.mkdir(parents=True, exist_ok=True)
            
            # Create audio filename
            output_filename = f"audio_{frame_index}.wav"
            output_path = video_audio_path / output_filename
            
            logger.info(f"Converting text to speech using {current_engine} for {folder_name}, frame {frame_index}")
            
            success = False
            voice_used = ""
            
            # Call appropriate TTS API based on selected engine
            if current_engine == TTS_ENGINE_OPENAI:
                success = await call_openai_api(self.client, output_path, OPENAI_MODEL_NAME, OPENAI_TTS_VOICE, text)
                voice_used = OPENAI_TTS_VOICE
            elif current_engine == TTS_ENGINE_GTTS:
                success = await call_gtts(output_path, GTTS_LANGUAGE, text, self.aiogTTS_engine)
                voice_used = f"aiogTTS ({GTTS_LANGUAGE})"
            elif current_engine == TTS_ENGINE_KOKORO:
                success = await call_kokoro(output_path, KOKORO_VOICE, KOKORO_SPEED, text, self.kokoro_pipeline)
                voice_used = KOKORO_VOICE
            else:
                logger.warning(f"Unknown engine {current_engine}, falling back to Kokoro")
                success = await call_kokoro(output_path, KOKORO_VOICE, KOKORO_SPEED, text, self.kokoro_pipeline)
                voice_used = OPENAI_TTS_VOICE
            
            # Get audio duration if file exists
            duration = None
            if output_path.exists():
                try:
                    with contextlib.closing(wave.open(str(output_path), 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)
                except Exception as e:
                    logger.warning(f"Could not get audio duration: {str(e)}")
            
            audio_response = AudioResponse(
                audio_path=str(output_path),
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
                audio_path="",
                text=text,
                engine=engine if engine else self.engine,
                voice="",
                format="wav"
            )