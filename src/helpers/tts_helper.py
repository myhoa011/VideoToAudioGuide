import asyncio
import soundfile as sf
from src.utils.logger import logger

# Module imports for different TTS engines
async def call_openai_api(openai_client, output_path, model, voice, text):
    """Use OpenAI API for text-to-speech conversion"""
    try:
        response = await asyncio.to_thread(
            openai_client.audio.speech.create,
            model=model,
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        await asyncio.to_thread(response.stream_to_file, output_path)
        return True
    except Exception as e:
        logger.error(f"Error in OpenAI TTS: {str(e)}")
        return False

async def call_gtts(output_path, language, text, agtts_engine=None):
    """Use aiogTTS (async Google Text-to-Speech) for conversion"""
    try:
        # Use provided engine or create new one
        agtts = agtts_engine
        
        # Use async save method
        await agtts.save(
            text=text,
            filename=str(output_path),
            lang=language
        )
        return True
    except Exception as e:
        logger.error(f"Error in aiogTTS: {str(e)}")
        return False

async def call_kokoro(output_path, voice, speed, text, kokoro_pipeline=None):
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
            sf.write(str(output_path), audio, 24000)
            return True
            
        return True
    except Exception as e:
        logger.error(f"Error in Kokoro TTS: {str(e)}")
        return False

