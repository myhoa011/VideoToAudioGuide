from openai import OpenAI
import os
from pathlib import Path
import asyncio

from src.utils.logger import logger
from src.helpers.openai_tts_helper import call_api
from src.utils.constant import OUTPUT_AUDIO_PATH, OPENAI_MODEL_NAME, OPENAI_TTS_VOICE
from src.initializer import initializer

class TextToSpeechHandler:
    """Handler for text-to-speech conversion using OpenAI API"""
    
    def __init__(self, model_name=OPENAI_MODEL_NAME, voice=OPENAI_TTS_VOICE):
        """Initialize text-to-speech handler"""
        self.client = initializer.get_openai_client()
        self.output_path = Path(OUTPUT_AUDIO_PATH)
        self.model = model_name
        self.voice = voice
    
    async def convert_text_to_speech(self, text: str, output_filename: str, voice: str = "alloy") -> str:
        """Convert text to speech and save as audio file
        
        Args:
            text (str): Text to convert to speech
            output_filename (str): Filename for the output audio
            voice (str, optional): Voice to use. Defaults to "alloy".
                Options: alloy, echo, fable, onyx, nova, shimmer
                
        Returns:
            str: Path to the saved audio file
        """
        try:
            # Ensure file has .mp3 extension
            if not output_filename.endswith('.mp3'):
                output_filename += '.mp3'
                
            output_path = self.output_path / output_filename
            
            # Call OpenAI API to convert text to speech
            audio_content = await call_api(self.client, voice, text)
            
            # Save audio content to file
            with open(output_path, 'wb') as audio_file:
                audio_file.write(audio_content)
                
            logger.info(f"Audio saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            raise