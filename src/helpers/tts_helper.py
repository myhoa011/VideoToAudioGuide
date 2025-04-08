import asyncio
import soundfile as sf
import io
import numpy as np
from src.utils.logger import logger
from src.utils.constant import AUDIO_SAMPLE_RATE, AUDIO_FORMAT

# Module imports for different TTS engines
async def call_openai_api(openai_client, model, voice, text):
    """Use OpenAI API for text-to-speech conversion"""
    try:
        response = await asyncio.to_thread(
            openai_client.audio.speech.create,
            model=model,
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        # Get bytes from response
        audio_bytes = await asyncio.to_thread(lambda: response.read())
        return audio_bytes
    except Exception as e:
        logger.error(f"Error in OpenAI TTS: {str(e)}")
        return None

async def call_gtts(language, text, agtts_engine=None):
    """Use aiogTTS (async Google Text-to-Speech) for conversion"""
    try:
        # Use provided engine or create new one
        agtts = agtts_engine
        
        # Create a BytesIO object
        fp = io.BytesIO()
        # Use async write_to_fp method
        await agtts.write_to_fp(
            text=text,
            fp=fp,
            lang=language
        )
        # Reset position to start
        fp.seek(0)
        # Get bytes
        audio_bytes = fp.read()
        return audio_bytes
    except Exception as e:
        logger.error(f"Error in aiogTTS: {str(e)}")
        return None

async def call_kokoro(voice, speed, text, kokoro_pipeline=None):
    """Use Kokoro TTS library for conversion"""
    try:
        # Use provided pipeline or create new one
        pipeline = kokoro_pipeline
        
        # Create generator
        generator = pipeline(
            text=text,
            voice=voice,
            speed=speed
        )
        
        # Process results from generator
        for i, (gs, ps, audio) in enumerate(generator):
            # Only take the first audio
            # Convert numpy array to bytes
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio, AUDIO_SAMPLE_RATE, format=AUDIO_FORMAT)
            audio_bytes.seek(0)
            return audio_bytes.read()
            
        return None
    except Exception as e:
        logger.error(f"Error in Kokoro TTS: {str(e)}")
        return None

