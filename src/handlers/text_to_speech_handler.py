from pathlib import Path
import os
from src.utils.logger import logger
from src.schemas.audio import AudioResponse
from src.utils.constant import OPENAI_MODEL_NAME, OPENAI_TTS_VOICE, OUTPUT_AUDIO_PATH
from src.helpers.openai_tts_helper import call_api
from src.initializer import initializer
import wave
import contextlib

class TextToSpeechHandler:
    """Handler for text-to-speech conversion"""
    
    def __init__(self, output_path: str = OUTPUT_AUDIO_PATH):
        self.client = initializer.get_openai_client()
        self.model = OPENAI_MODEL_NAME
        self.voice = OPENAI_TTS_VOICE
        self.base_output_path = Path(output_path)
        self.base_output_path.mkdir(parents=True, exist_ok=True)
    
    async def convert_text_to_speech(self, text: str, folder_name: str, frame_index: str) -> AudioResponse:
        """
        Convert text to speech using OpenAI API and save in folder structure like frames
        
        Args:
            text: Text to convert
            folder_name: Video folder name
            frame_index: Frame index
            
        Returns:
            AudioResponse: Audio file information
        """
        try:
            # Create a folder for this video's audio files
            video_audio_path = self.base_output_path / folder_name
            video_audio_path.mkdir(parents=True, exist_ok=True)
            
            # Create audio filename
            output_filename = f"audio_{frame_index}.wav"
            output_path = video_audio_path / output_filename
            
            logger.info(f"Converting text to speech for {folder_name}, frame {frame_index}")
            
            await call_api(self.client, output_path, self.model, self.voice, text)
            
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
                voice=self.voice,
                format='wav'
            )
            
            return audio_response
                
        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise