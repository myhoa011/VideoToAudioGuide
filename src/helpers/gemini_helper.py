from google.genai import types
from PIL import Image
from io import BytesIO
import aiofiles
import asyncio
from typing import List
from src.utils.logger import logger
from schemas import DetectedObject

async def call_api(gemini_client, prompt, system_instructions, safety_settings, model_name, img_path: str) -> List[DetectedObject]:
    """
    Call Gemini API to detect objects in an image.

    Args:
        gemini_client: Gemini API client instance.
        prompt (str): Prompt for the model.
        system_instructions: System instructions for the model.
        safety_settings: Safety settings for content generation.
        model_name (str): Name of the Gemini model.
        img_path (str): Path to the image file.

    Returns:
        List[DetectedObject]: List of detected objects with bounding boxes.
    """
    try:
        # Read image file
        async with aiofiles.open(img_path, mode='rb') as file:
            content = await file.read()
            img = Image.open(BytesIO(content))

        # Call Gemini API
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model_name,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                temperature=0,
                safety_settings=safety_settings,
                response_mime_type="application/json",
                response_schema=list[DetectedObject]
            )
        )

        detected_objects: list[DetectedObject] = response.parsed

        if detected_objects:
            logger.info(f"Gemini API detected {len(detected_objects)} objects.")
            return detected_objects
        else:
            logger.warning("Gemini API response parsed successfully but no objects detected")
            return []

    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return []